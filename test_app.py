import csv
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
        self.assertFalse(app.multiselect)

        app.button[0].click().run()
        self.assertTrue(app.warning)

        app.selectbox[1].set_value(1).run()
        with ruta.with_name("Pacientes.csv").open(encoding="utf-8-sig", newline="") as archivo:
            antecedentes_csv = {
                fila["antecedente_especifico"].strip()
                for fila in csv.DictReader(archivo, delimiter=";")
                if fila["antecedentes_relevantes"] == "1"
                and fila["antecedente_especifico"].strip().casefold() not in ("", "ninguno")
            }
        self.assertGreater(len(antecedentes_csv), 6)
        self.assertEqual(set(app.multiselect[0].options), antecedentes_csv)
        seleccion = app.multiselect[0].options[:2]
        app.multiselect[0].set_value(seleccion)

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
                self.assertIn("Recorrido del árbol", [titulo.value for titulo in app.subheader])
                self.assertTrue(any(
                    "Árbol: Nivel " in salida.value and "Red neuronal: Nivel " in salida.value
                    for salida in app.markdown
                ))

        for antecedente in seleccion:
            self.assertIn(f"- {antecedente}", [salida.value for salida in app.markdown])
        app.selectbox[1].set_value(0).run()
        self.assertFalse(app.multiselect)


if __name__ == "__main__":
    unittest.main()
