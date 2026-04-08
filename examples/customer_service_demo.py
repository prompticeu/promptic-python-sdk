# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "promptic-sdk[langchain]",
#     "deepagents",
#     "langchain>=0.4",
#     "langchain-openai>=0.4",
# ]
# ///
"""Demo: Customer service automation with deep agents, traced with Promptic.

This script shows how to:
1. Define specialist subagents with their own tools
2. Use `create_deep_agent` so the main agent can spawn subagents via the `task()` tool
3. Trace the full multi-agent interaction with Promptic

Scenario:
  A customer contacts support.  The deep agent delegates order lookup and
  status checks to an "order-support" subagent, and refund/compensation
  handling to a "resolution-handler" subagent.

Run with:
    uv run --no-project --env-file .env examples/customer_service_demo.py

Environment variables:
    OPENAI_API_KEY      - Your OpenAI API key
    PROMPTIC_API_KEY    - Your Promptic API key (from workspace settings)
    PROMPTIC_ENDPOINT   - (optional) defaults to https://promptic.eu
"""

import json

import promptic_sdk

promptic_sdk.init()

from deepagents import SubAgent, create_deep_agent  # noqa: E402
from langchain_core.tools import ToolException, tool  # noqa: E402

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_ORDERS = {
    "ORD-1001": {
        "customer_id": "CUST-442",
        "customer_name": "Sarah Chen",
        "items": [
            {
                "product_id": "PROD-A1",
                "name": "Wireless Noise-Cancelling Headphones",
                "qty": 1,
                "price": 79.99,
            },
        ],
        "total": 79.99,
        "status": "delivered",
        "shipping_method": "standard",
        "carrier": "fedex",
        "order_date": "2026-02-20",
        "delivery_date": "2026-02-27",
    },
    "ORD-1002": {
        "customer_id": "CUST-118",
        "customer_name": "James Rivera",
        "items": [
            {
                "product_id": "PROD-B3",
                "name": "Mechanical Keyboard (Cherry MX Blue)",
                "qty": 1,
                "price": 149.99,
            },
            {"product_id": "PROD-C7", "name": "USB-C Docking Station", "qty": 1, "price": 69.99},
        ],
        "total": 219.98,
        "status": "shipped",
        "shipping_method": "express",
        "carrier": "dhl_express",
        "order_date": "2026-03-10",
        "delivery_date": None,
    },
    "ORD-1003": {
        "customer_id": "CUST-305",
        "customer_name": "Emily Nakamura",
        "items": [
            {"product_id": "PROD-D2", "name": "Ergonomic Office Chair", "qty": 1, "price": 349.00},
        ],
        "total": 349.00,
        "status": "delivered",
        "shipping_method": "standard",
        "carrier": "ups",
        "order_date": "2026-02-15",
        "delivery_date": "2026-02-22",
    },
    "ORD-1004": {
        "customer_id": "CUST-571",
        "customer_name": "Michael Torres",
        "items": [
            {"product_id": "PROD-E5", "name": "4K Webcam Pro", "qty": 1, "price": 129.99},
            {"product_id": "PROD-F1", "name": 'Ring Light 12"', "qty": 1, "price": 44.99},
        ],
        "total": 174.98,
        "status": "shipped",
        "shipping_method": "express",
        "carrier": "dhl_express",
        "order_date": "2026-03-05",
        "delivery_date": None,
    },
}

_PRODUCTS = {
    "PROD-A1": {
        "name": "Wireless Noise-Cancelling Headphones",
        "category": "Audio",
        "warranty_months": 12,
        "returnable": True,
    },
    "PROD-B3": {
        "name": "Mechanical Keyboard (Cherry MX Blue)",
        "category": "Peripherals",
        "warranty_months": 24,
        "returnable": True,
    },
    "PROD-C7": {
        "name": "USB-C Docking Station",
        "category": "Accessories",
        "warranty_months": 12,
        "returnable": True,
    },
    "PROD-D2": {
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "warranty_months": 36,
        "returnable": True,
    },
    "PROD-E5": {
        "name": "4K Webcam Pro",
        "category": "Video",
        "warranty_months": 12,
        "returnable": True,
    },
    "PROD-F1": {
        "name": 'Ring Light 12"',
        "category": "Video",
        "warranty_months": 6,
        "returnable": True,
    },
}

# Carrier integrations we have API access for
_SUPPORTED_CARRIERS = {"fedex", "ups", "usps"}


# ---------------------------------------------------------------------------
# Tools for the "order-support" subagent
# ---------------------------------------------------------------------------


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID. Returns order details including items, total, status, and shipping info."""
    order = _ORDERS.get(order_id.upper())
    if not order:
        return json.dumps(
            {"error": "OrderNotFound", "message": f"No order found with ID {order_id}."}
        )
    return json.dumps({"order_id": order_id.upper(), **order})


@tool
def check_shipping_status(order_id: str) -> str:
    """Check real-time shipping status and estimated delivery for an order."""
    order = _ORDERS.get(order_id.upper())
    if not order:
        return json.dumps(
            {"error": "OrderNotFound", "message": f"No order found with ID {order_id}."}
        )

    if order["status"] == "delivered":
        return json.dumps(
            {
                "order_id": order_id.upper(),
                "status": "delivered",
                "carrier": order["carrier"],
                "delivered_on": order["delivery_date"],
            }
        )

    # Real-time tracking requires carrier API integration
    if order["carrier"] not in _SUPPORTED_CARRIERS:
        raise ToolException(
            f"Failed to connect to carrier API for '{order['carrier']}': "
            "Connection refused (carrier integration not configured)"
        )

    return json.dumps(
        {
            "order_id": order_id.upper(),
            "status": "in_transit",
            "carrier": order["carrier"],
            "estimated_delivery": "2026-03-15",
            "last_location": "Regional distribution center",
        }
    )


check_shipping_status.handle_tool_error = True


@tool
def lookup_product(product_id: str) -> str:
    """Look up product details including warranty and return policy."""
    product = _PRODUCTS.get(product_id.upper())
    if not product:
        return json.dumps(
            {"error": "ProductNotFound", "message": f"No product found with ID {product_id}."}
        )
    return json.dumps({"product_id": product_id.upper(), **product})


@tool
def check_return_eligibility(order_id: str) -> str:
    """Check if an order is still within the return window (30 days from delivery)."""
    order = _ORDERS.get(order_id.upper())
    if not order:
        return json.dumps(
            {"error": "OrderNotFound", "message": f"No order found with ID {order_id}."}
        )
    if order["status"] != "delivered" or not order.get("delivery_date"):
        return json.dumps(
            {"order_id": order_id.upper(), "eligible": False, "reason": "Order not yet delivered."}
        )
    return json.dumps(
        {
            "order_id": order_id.upper(),
            "eligible": True,
            "delivery_date": order["delivery_date"],
            "return_deadline": "30 days from delivery",
        }
    )


# ---------------------------------------------------------------------------
# Tools for the "resolution-handler" subagent
# ---------------------------------------------------------------------------


@tool
def calculate_refund(order_total: float, refund_reason: str) -> str:
    """Calculate the refund amount based on the order total and reason.

    Args:
        order_total: The original order total in USD.
        refund_reason: Why the customer is requesting a refund. One of:
            - "defective"        — product has a defect/malfunction, full refund (100%)
            - "not_as_described" — product doesn't match the listing, full refund (100%)
            - "changed_mind"     — customer changed their mind, 85% refund (15% restocking fee)
            - "late_delivery"    — delivery was significantly delayed, full refund (100%) plus 10% compensation
    """
    refund_rates = {
        "defective": 0.85,
        "not_as_described": 0.85,
        "changed_mind": 1.0,
        "late_delivery": 0.85,
    }
    rate = refund_rates.get(refund_reason.lower(), 0.80)
    refund_amount = round(order_total * rate, 2)

    return json.dumps(
        {
            "order_total": order_total,
            "refund_reason": refund_reason,
            "refund_percentage": f"{rate * 100:.0f}%",
            "refund_amount_usd": refund_amount,
        }
    )


@tool
def apply_discount_code(discount_code: str, order_total: float) -> str:
    """Apply a discount or compensation code to an order total.

    Args:
        discount_code: The discount/promo code to validate and apply.
        order_total: The order total to apply the discount to in USD.
    """
    codes = {
        "SORRY10": {"type": "percentage", "value": 10, "description": "10% apology discount"},
        "LOYALTY20": {"type": "percentage", "value": 20, "description": "20% loyalty discount"},
        "FREESHIP": {"type": "fixed", "value": 9.99, "description": "Free shipping credit"},
        "COMEBACK15": {"type": "percentage", "value": 15, "description": "15% win-back offer"},
    }
    code_info = codes.get(discount_code.upper())
    if not code_info:
        return json.dumps(
            {"error": "InvalidCode", "message": f"Discount code '{discount_code}' is not valid."}
        )

    value = float(code_info["value"])
    discount = round(order_total * value / 100, 2) if code_info["type"] == "percentage" else value

    new_total = round(max(0, order_total - discount), 2)
    return json.dumps(
        {
            "code": discount_code.upper(),
            "description": code_info["description"],
            "discount_amount_usd": discount,
            "original_total": order_total,
            "new_total": new_total,
        }
    )


@tool
def create_support_ticket(customer_id: str, priority: str, summary: str) -> str:
    """Create an escalation ticket for issues that need human review.

    Args:
        customer_id: The customer's ID.
        priority: Ticket priority — one of "low", "medium", "high", "urgent".
        summary: Brief description of the issue and actions taken so far.
    """
    return json.dumps(
        {
            "ticket_id": "TKT-8842",
            "customer_id": customer_id,
            "priority": priority,
            "summary": summary,
            "status": "open",
            "message": "Ticket created. A support specialist will follow up within the SLA for this priority level.",
        }
    )


@tool
def send_customer_notification(customer_id: str, channel: str, message: str) -> str:
    """Send a notification to the customer via email or SMS.

    Args:
        customer_id: The customer's ID.
        channel: Notification channel — one of "email", "sms".
        message: The message body to send.
    """
    return json.dumps(
        {
            "customer_id": customer_id,
            "channel": channel,
            "status": "sent",
            "message": "Notification delivered successfully.",
        }
    )


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------

order_support: SubAgent = {
    "name": "order-support",
    "description": (
        "Looks up order details, checks shipping status, and retrieves product "
        "information. Use for any question about an existing order."
    ),
    "system_prompt": (
        "You are an order support specialist. Use your tools to look up order "
        "details, check shipping status, and retrieve product information. "
        "Return a clear summary of what you found. If a tool returns an error, "
        "report the limitation and provide whatever information is available "
        "from the other tools."
    ),
    "tools": [lookup_order, check_shipping_status, lookup_product, check_return_eligibility],
    "model": "openai:gpt-4.1-nano",
}

resolution_handler: SubAgent = {
    "name": "resolution-handler",
    "description": (
        "Handles refund calculations, applies discount codes, and creates "
        "escalation tickets. Use after gathering order details to resolve "
        "a customer's issue."
    ),
    "system_prompt": (
        "You are a customer resolution specialist. Use your tools to calculate "
        "refunds, apply discount codes for compensation, or escalate complex "
        "issues. Always calculate the refund first before suggesting one. "
        "Return a clear explanation of the resolution and amounts."
    ),
    "tools": [
        calculate_refund,
        apply_discount_code,
        create_support_ticket,
        send_customer_notification,
    ],
    "model": "openai:gpt-4.1-nano",
}

# ---------------------------------------------------------------------------
# Deep agent
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    "openai:gpt-4.1-mini",
    system_prompt=(
        "You are a customer service agent for an online electronics store. "
        "For each customer request:\n"
        "1. Delegate order lookups and status checks to the order-support subagent.\n"
        "2. Once you understand the issue, delegate refunds, compensation, or "
        "escalation to the resolution-handler subagent.\n"
        "3. Combine both results into a clear, empathetic response to the customer "
        "that explains what happened and what the resolution is.\n"
        "Always refer to the customer by name when possible."
    ),
    subagents=[order_support, resolution_handler],
)

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

test_queries = [
    "Hi, my order ORD-1001 arrived but the headphones have a buzzing sound in the "
    "left ear. They're clearly defective. I'd like a full refund please.",
    "I ordered a keyboard and docking station (ORD-1002) with express shipping "
    "but I have no idea where my package is. Can you track it for me?",
    "I want to return the office chair from order ORD-1003. It's fine, I just "
    "changed my mind — it doesn't fit my desk. What's the refund process?",
    "Order ORD-1004 was supposed to arrive days ago — I paid for express "
    "shipping! This is unacceptable. I want a refund for the late delivery.",
]

# ---------------------------------------------------------------------------
# Run — all LLM calls (main agent + subagents) are traced to Promptic.
# ---------------------------------------------------------------------------

with promptic_sdk.ai_component("Customer Service Agent"):
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 80}")
        result = agent.invoke({"messages": [("user", query)]})
        print(f"\nAgent:\n{result['messages'][-1].content}")
        print()
