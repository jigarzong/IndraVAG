"""Directorio de usuarios DEMO. En produccion la identidad la emite el proveedor corporativo (SSO/Entra ID) y esta tabla no existe:
el API solo valida el JWT. AUTH_MODE=demo (por defecto en el MVP) habilita POST /auth/token para estos usuarios; cualquier otro valor lo apaga."""

DEMO_USERS = {
    "ana.supervisor": dict(role="supervisor", line_id="L1", name="Ana (supervisora L1)"),
    "luis.supervisor": dict(role="supervisor", line_id="L1", name="Luis (supervisor L1)"),
    "carla.calidad": dict(role="calidad", line_id="L1", name="Carla (calidad L1)"),
    "mario.mantenimiento": dict(role="mantenimiento", line_id="L1", name="Mario (mantenimiento L1)"),
    "vega.ingeniero_ml": dict(role="ingeniero_ml", line_id="L1", name="Vega (ingeniera ML L1)"),
    "pablo.ingeniero_ml": dict(role="ingeniero_ml", line_id="L1", name="Pablo (ingeniero ML L1)"),
    "rosa.supervisor_l2": dict(role="supervisor", line_id="L2", name="Rosa (supervisora L2)"),
}
