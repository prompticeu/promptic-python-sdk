def score(row):
    """Check that the extracted line items add up to the invoice total."""
    output = row["output"] or {}
    items = output.get("line_items") or []
    total = sum(item.get("amount", 0) for item in items)
    if abs(total - output.get("total", 0)) < 0.01:
        return {"score": 1.0, "reason": "Line items sum to the total"}
    return {"score": 0.0, "reason": f"Line items sum to {total:.2f}"}
