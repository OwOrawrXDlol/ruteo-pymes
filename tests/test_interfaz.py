"""Pruebas de las piezas comunes de la interfaz (interfaz.py): formatos, nombres y colores."""

from interfaz import COLORES, NOMBRES, dinero, dinero_centavos, etiqueta, km, pct, tonos


def test_formatos_iguales_en_toda_la_app():
    assert dinero(129384.79) == "$129,385"
    assert dinero_centavos(12.345) == "$12.35"
    assert km(14895.9) == "14,896 km" and km(12.34, 1) == "12.3 km"
    # Guion normal como signo menos: st.metric lo usa para pintar la flecha del color correcto
    assert pct(-16.44) == "-16.4 %" and pct(5) == "+5.0 %" and pct(58.73, signo=False) == "58.7 %"


def test_nombres_humanos_con_el_codigo_como_etiqueta():
    assert etiqueta("E1") == "Como se hace hoy (E1)"
    assert NOMBRES["E3"] == "Pronóstico + rutas (propuesta)"
    assert set(COLORES) == set(NOMBRES)


def test_tonos_del_color_del_escenario():
    assert tonos("E3", 1) == [COLORES["E3"].lower()]
    varios = tonos("E3", 5)
    assert len(set(varios)) == 5
    brillo = [sum(int(c[k:k + 2], 16) for k in (1, 3, 5)) for c in varios]
    assert brillo == sorted(brillo)  # de oscuro a claro
