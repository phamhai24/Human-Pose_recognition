from pydantic import BaseModel


class Health(BaseModel):
    status: str
    message: str


def error(code, message):
    return {'type': 'error', 'code': code, 'message': message}
