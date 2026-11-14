# ----------------- FOTOAPP -----------------
# Procesamiento de imágenes con Python

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps


# ----------------- CONFIGURACIÓN -----------------
# Dimensiones recomendadas para cada red social
DIMENSIONES = {
    "youtube": (1280, 720),
    "instagram": (1080, 1350),
    "twitter": (1600, 900),
    "facebook": (1200, 630),}

# Filtros disponibles en el menú
FILTROS_MENU = {
    "1": "blur",
    "2": "contour",
    "3": "detail",
    "4": "edge enhance",
    "5": "edge enhance more",
    "6": "emboss",
    "7": "find edges",
    "8": "sharpen",
    "9": "smooth",}

FILTROS_PILLOW = {
    "blur": ImageFilter.BLUR,
    "contour": ImageFilter.CONTOUR,
    "detail": ImageFilter.DETAIL,
    "edge enhance": ImageFilter.EDGE_ENHANCE,
    "edge enhance more": ImageFilter.EDGE_ENHANCE_MORE,
    "emboss": ImageFilter.EMBOSS,
    "find edges": ImageFilter.FIND_EDGES,
    "sharpen": ImageFilter.SHARPEN,
    "smooth": ImageFilter.SMOOTH,}


# ----------------- FUNCIONES AUXILIARES -----------------

def validar_imagen(imagen):
    """Comprueba que se haya recibido una imagen válida de Pillow."""
    if imagen is None:
        raise ValueError(
            "No hay una imagen cargada. Primero debe utilizar la opción 1.")
    if not isinstance(imagen, Image.Image):
        raise TypeError("La imagen debe ser una instancia de PIL.Image.Image.")


def es_entorno_interactivo():
    """Indica si el programa se está ejecutando en Jupyter o Google Colab."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def limpiar_pantalla():
    """Limpia la terminal. En notebooks conserva las salidas anteriores."""
    if es_entorno_interactivo():
        return
    os.system("cls" if os.name == "nt" else "clear")


def mostrar_comparacion(
    imagen_original,
    imagen_resultado,
    titulo_resultado,
    guardar_en=None,
    mapa_color=None,):
    """Muestra y, si se solicita, guarda una comparación de dos imágenes."""
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5))

    ejes[0].imshow(imagen_original)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(imagen_resultado, cmap=mapa_color)
    ejes[1].set_title(titulo_resultado)
    ejes[1].axis("off")

    figura.tight_layout()
    if guardar_en is not None:
        ruta = Path(guardar_en)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(ruta, dpi=150, bbox_inches="tight")

    # En terminal, la ventana no bloquea el menú.
    # En Colab/Jupyter, show() presenta la figura dentro de la celda.
    if es_entorno_interactivo():
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.1)
    return figura


def pausar_y_cerrar_figuras():
    """Espera la confirmación en terminal y cierra las figuras."""
    if not es_entorno_interactivo():
        input("\nPresione Enter para volver al menú...")

    # En terminal cierra las ventanas de Matplotlib después de la pausa.
    # En Colab la imagen ya quedó mostrada como salida de la celda.
    plt.close("all")


# ----------------- (1) CARGAR Y REDIMENSIONAR -----------------
def abrir_imagen(ruta):
    """Abre una imagen desde una ruta local o URL y la convierte a RGB."""
    ruta = str(ruta).strip()
    if not ruta:
        raise ValueError("Debe ingresar una ruta o URL de imagen.")
    if ruta.startswith(("http://", "https://")):
        peticion = Request(ruta, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(peticion, timeout=15) as respuesta:
                datos = respuesta.read()
            with Image.open(BytesIO(datos)) as imagen:
                return imagen.convert("RGB")
        except (HTTPError, URLError, OSError) as error:
            raise FileNotFoundError(
                f"No se pudo descargar o abrir la imagen desde la URL: {error}") from error
    ruta_local = Path(ruta)
    if not ruta_local.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de imagen: {ruta_local}")
    try:
        with Image.open(ruta_local) as imagen:
            return imagen.convert("RGB")
    except OSError as error:
        raise ValueError(f"El archivo indicado no es una imagen válida: {error}") from error


def leer_y_redimensionar(ruta, palabra_clave, guardar_en=None):
    """
    Redimensiona una imagen según la plataforma elegida, conserva su
    proporción y la centra sobre un lienzo blanco.
    """
    plataforma = palabra_clave.lower().strip()
    if plataforma not in DIMENSIONES:
        plataformas = ", ".join(DIMENSIONES.keys())
        raise ValueError(f"Plataforma no válida. Opciones: {plataformas}.")
    imagen = abrir_imagen(ruta)
    ancho, alto = DIMENSIONES[plataforma]
    imagen_ajustada = ImageOps.contain(imagen, (ancho, alto), method=Image.Resampling.LANCZOS,)
    lienzo = Image.new("RGB", (ancho, alto), "white")
    posicion = (
        (ancho - imagen_ajustada.width) // 2,
        (alto - imagen_ajustada.height) // 2,)
    lienzo.paste(imagen_ajustada, posicion)
    if guardar_en is not None:
        ruta_guardado = Path(guardar_en)
        ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
        lienzo.save(ruta_guardado)
    return lienzo


# ----------------- (2) AJUSTAR CONTRASTE -----------------
def ajustar_contraste_histograma(imagen, guardar_en=None):
    """
    Ecualiza el histograma del canal de luminosidad en espacio YCrCb,
    conservando los canales de crominancia.
    """
    validar_imagen(imagen)
    imagen_array = np.array(imagen.convert("RGB"))
    imagen_ycrcb = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(imagen_ycrcb)
    y_ecualizado = cv2.equalizeHist(y)
    imagen_ecualizada = cv2.merge((y_ecualizado, cr, cb))
    imagen_ecualizada = cv2.cvtColor(imagen_ecualizada, cv2.COLOR_YCrCb2RGB,)
    resultado = Image.fromarray(imagen_ecualizada)
    mostrar_comparacion(
        imagen,
        resultado,
        "Imagen con contraste ajustado",
        guardar_en,)
    return resultado


# ----------------- (3) APLICAR FILTRO -----------------
def aplicar_filtro(imagen, nombre_filtro, guardar_en=None):
    """Aplica un filtro de Pillow y muestra la comparación con el original."""
    validar_imagen(imagen)
    nombre = nombre_filtro.lower().strip()
    if nombre not in FILTROS_PILLOW:
        opciones = ", ".join(FILTROS_PILLOW.keys())
        raise ValueError(f"Filtro no válido. Opciones: {opciones}.")
    resultado = imagen.filter(FILTROS_PILLOW[nombre])
    mostrar_comparacion(
        imagen,
        resultado,
        f"Filtro: {nombre.title()}",
        guardar_en,)
    return resultado


# ----------------- (4) DETECTAR BORDES Y CONTORNOS -----------------
def detectar_bordes(imagen, tecnica="canny", guardar_en=None):
    """Detecta bordes mediante Sobel o Canny y muestra la comparación."""
    validar_imagen(imagen)
    gris = cv2.cvtColor(
        np.array(imagen.convert("RGB")),
        cv2.COLOR_RGB2GRAY,)
    tecnica = tecnica.lower().strip()
    if tecnica == "sobel":
        gradiente_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        gradiente_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitud = np.sqrt(gradiente_x**2 + gradiente_y**2)
        bordes = cv2.convertScaleAbs(magnitud)
    elif tecnica == "canny":
        bordes = cv2.Canny(gris, 100, 200)
    else:
        raise ValueError("Técnica no válida. Debe elegir Sobel o Canny.")
    resultado = Image.fromarray(bordes)
    mostrar_comparacion(
        imagen,
        resultado,
        f"Detección de bordes: {tecnica.title()}",
        guardar_en,
        mapa_color="gray",)
    return resultado


# ----------------- (5) MENÚ PRINCIPAL -----------------
def main(carpeta_resultados="Resultados"):
    """
    Ejecuta el menú de FOTOAPP. La imagen cargada se conserva mientras
    el menú permanece en ejecución.
    """
    carpeta_resultados = Path(carpeta_resultados)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    imagen_actual = None
    while True:
        limpiar_pantalla()
        print("\n========== FOTOAPP ==========")
        print("1. Cargar y redimensionar imagen")
        print("2. Ajustar contraste")
        print("3. Aplicar filtro")
        print("4. Detectar bordes y contornos")
        print("5. Salir")
        try:
            opcion = input("\nSeleccione una opción: ").strip()
        except EOFError:
            print("\nNo se recibió una opción. Se finaliza FOTOAPP.")
            break
        if opcion == "1":
            ruta = input("Ingrese la ruta local o URL de la imagen: ").strip()
            plataforma = input(
                "Ingrese la plataforma (YouTube, Instagram, Twitter o Facebook): ").strip().lower()
            try:
                nombre_archivo = f"redimensionada_{plataforma}.png"
                ruta_guardado = carpeta_resultados / nombre_archivo

                # Se reemplaza la imagen actual solo si la carga fue exitosa.
                nueva_imagen = leer_y_redimensionar(
                    ruta,
                    plataforma,
                    ruta_guardado,)
                imagen_actual = nueva_imagen
                print(
                    f"\nImagen cargada y redimensionada correctamente."
                    f"\nGuardada en: {ruta_guardado}")
            except (FileNotFoundError, ValueError, TypeError, OSError) as error:
                print(f"\nError al cargar la imagen: {error}")
        elif opcion in ("2", "3", "4"):
            if imagen_actual is None:
                print("\nPrimero debe cargar una imagen utilizando la opción 1.")
            else:
                try:
                    if opcion == "2":
                        ruta_guardado = carpeta_resultados / "contraste_menu.png"
                        ajustar_contraste_histograma(imagen_actual, ruta_guardado,)
                        print(f"\nComparación guardada en: {ruta_guardado}")
                    elif opcion == "3":
                        print("\nFiltros disponibles:")
                        for numero, nombre in FILTROS_MENU.items():
                            print(f"{numero}. {nombre.title()}")
                        seleccion = input(
                            "\nSeleccione el número o nombre del filtro: ").strip().lower()
                        filtro = FILTROS_MENU.get(seleccion, seleccion)
                        nombre_archivo = filtro.replace(" ", "_")
                        ruta_guardado = (
                            carpeta_resultados / f"filtro_{nombre_archivo}.png")
                        aplicar_filtro(imagen_actual,filtro, ruta_guardado,)
                        print(f"\nComparación guardada en: {ruta_guardado}")
                    elif opcion == "4":
                        tecnica = input(
                            "Ingrese la técnica (Sobel o Canny): ").strip().lower()
                        ruta_guardado = (
                            carpeta_resultados / f"bordes_{tecnica}.png")
                        detectar_bordes(imagen_actual,tecnica,ruta_guardado,)
                        print(f"\nComparación guardada en: {ruta_guardado}")
                except (ValueError, TypeError, OSError, cv2.error) as error:
                    print(f"\nNo se pudo completar la operación: {error}")
        elif opcion == "5":
            print("\nGracias por utilizar FOTOAPP. ¡Hasta luego!")
            break
        else:
            print("\nOpción no válida. Seleccione un número del 1 al 5.")
        pausar_y_cerrar_figuras()


# Ejecutar el menú solamente cuando se inicia este archivo directamente.
if __name__ == "__main__":
    main("Resultados")
