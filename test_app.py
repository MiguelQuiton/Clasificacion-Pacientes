import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class ClasificacionTests(unittest.TestCase):
    def test_clasificacion_con_descripcion_libre(self):
        ruta = Path(__file__).with_name("app.py")
        app = AppTest.from_file(str(ruta), default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual([boton.label for boton in app.button], ["Clasificar"])

        app.button[0].click().run()
        self.assertTrue(app.warning)

        for cantidad, descripcion in (
            (1, "Dolor leve en la muñeca"),
            (4, "Dificultad para respirar"),
        ):
            with self.subTest(descripcion=descripcion):
                app.selectbox[0].set_value(cantidad)
                app.text_area[0].set_value(descripcion)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertFalse(app.radio)
                self.assertTrue(any(
                    "Clasificación estimada: Nivel " in salida.value
                    for salida in app.markdown
                ))


if __name__ == "__main__":
    unittest.main()
