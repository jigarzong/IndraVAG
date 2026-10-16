"""Identidad: JWT HS256. La identidad (usuario, rol, linea) la fija quien emite el token, nunca el LLM.

En el MVP el token lo emite la UI/API tras el inicio de sesion; el orquestador lo pasa al servidor MCP por entorno (stdio) o
por encabezado Authorization (HTTP). Ningun argumento de tool puede cambiar el usuario, el rol ni la linea.
"""
import time
from dataclasses import dataclass

import jwt

ROLES = ("supervisor", "calidad", "mantenimiento", "ingeniero_ml")


class AuthError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: str
    line_id: str
    plant_id: str
    session_id: str


def issue_token(secret, user_id, role, line_id, session_id, plant_id="P1", ttl_s=3600):
    if not secret:
        raise AuthError("JWT_SECRET no configurado")
    if role not in ROLES:
        raise AuthError(f"rol desconocido: {role}")
    now = int(time.time())
    return jwt.encode({"sub": user_id, "role": role, "line_id": line_id, "plant_id": plant_id, "sid": session_id,
                       "iat": now, "exp": now + ttl_s}, secret, algorithm="HS256")


def verify_token(secret, token):
    if not secret:
        raise AuthError("JWT_SECRET no configurado")
    if not token:
        raise AuthError("falta el token de sesion")
    try:
        c = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["sub", "role", "line_id", "sid", "exp"]})
    except jwt.ExpiredSignatureError:
        raise AuthError("token vencido")
    except jwt.PyJWTError as e:
        raise AuthError(f"token invalido ({type(e).__name__})")
    if c["role"] not in ROLES:
        raise AuthError("rol invalido en el token")
    return Identity(c["sub"], c["role"], c["line_id"], c.get("plant_id", "P1"), c["sid"])
