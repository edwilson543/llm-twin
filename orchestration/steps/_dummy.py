import logging

import zenml


@zenml.step
def get_user() -> str:
    username = "edwilson543"
    logging.info(f"Got user '{username}'")
    return username


@zenml.step
def do_something_with_user(username: str) -> None:
    logging.info(f"Did something with user '{username}'")
