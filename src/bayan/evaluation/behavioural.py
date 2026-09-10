"""Lab 6: behavioural test generators and runners."""


def run_behavioural_suite(predict_fn, tests):
    """
    Run behavioural tests.

    Expected test format:
    {
        "type": "invariance" | "directional" | "mft",
        "input": ...,
        ...
    }

    Returns overall pass rates by test type.
    """

    results = {
        "invariance": [],
        "directional": [],
        "mft": [],
    }

    for test in tests:
        test_type = test["type"]

        if test_type == "invariance":
            original = predict_fn(test["input"])
            changed = predict_fn(test["perturbed"])

            passed = original == changed

        elif test_type == "directional":
            before = predict_fn(test["input"])
            after = predict_fn(test["perturbed"])

            direction = test.get("direction", "increase")

            if direction == "increase":
                passed = after > before
            else:
                passed = after < before

        elif test_type == "mft":
            prediction = predict_fn(test["input"])
            passed = prediction == test["expected"]

        else:
            raise ValueError(
                f"Unknown behavioural test type: {test_type}"
            )

        results[test_type].append(bool(passed))

    summary = {}

    for name, values in results.items():
        if values:
            summary[name] = {
                "passed": sum(values),
                "total": len(values),
                "rate": sum(values) / len(values),
            }
        else:
            summary[name] = {
                "passed": 0,
                "total": 0,
                "rate": None,
            }

    return summary