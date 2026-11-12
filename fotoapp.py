# ----------------- FOTOAPP -----------------
# Importamos librerias
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ----------------- (1) -----------------
# Ancho y alto en píxeles de cada app
dimensiones = {
    "youtube": (1280, 720),
    "instagram": (1080, 1350),
    "twitter": (1600, 900),
    "facebook": (1200, 630)}

# Función para abrir la imagen desde una ruta local
def abrir_imagen(ruta):
    """Abre una imagen desde una ruta local y la convierte al formato RGB."""
    if not Path(ruta).exists():
        raise FileNotFoundError("No se encontró la imagen.")
    imagen = Image.open(ruta)
    return imagen.convert("RGB")

# Función principal de redimensionamiento
def leer_y_redimensionar(ruta, palabra_clave, guardar_en=None):
    """ Redimensiona una imagen según las dimensiones de la plataforma elegida,
    manteniendo su proporción original y centrando el resultado en un lienzo."""
    # Convertimos la palabra clave a minúsculas
    palabra_clave = palabra_clave.lower().strip()

    # Verificamos que la plataforma sea válida
    if palabra_clave not in dimensiones:
        raise ValueError("Plataforma no válida")

    # Abrimos la imagen
    imagen = abrir_imagen(ruta)

    # Obtenemos las dimensiones correspondientes
    ancho, alto = dimensiones[palabra_clave]

    # Redimensionamos manteniendo la proporción original
    imagen_ajustada = ImageOps.contain(
        imagen,
        (ancho, alto),
        method=Image.Resampling.LANCZOS)
    
    # Creamos un lienzo con las dimensiones finales
    lienzo = Image.new("RGB", (ancho, alto), "white")

    # Centramos la imagen dentro del lienzo
    posicion = (
        (ancho - imagen_ajustada.width) // 2,
        (alto - imagen_ajustada.height) // 2)
    lienzo.paste(imagen_ajustada, posicion)
    if guardar_en:
        lienzo.save(guardar_en)
    return lienzo
  

# ----------------- (2) -----------------
# Función para ajustar el contraste
def ajustar_contraste_histograma(imagen, guardar_en=None):
    """Mejora el contraste de una imagen mediante la ecualización del histograma
    del canal de luminosidad. Muestra y guarda una comparación entre la
    imagen original y la ecualizada."""
    # Convertimos la imagen Pillow en un array de NumPy
    imagen_array = np.array(imagen)

    # Convertimos la imagen de RGB a YCrCb
    imagen_ycrcb = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2YCrCb)

    # Separamos los canales de la imagen
    y, cr, cb = cv2.split(imagen_ycrcb)

    # Ecualizamos el histograma del canal de luminosidad
    y_ecualizado = cv2.equalizeHist(y)

    # Unimos nuevamente los canales y convertimos de YCrCb a RGB
    imagen_ecualizada = cv2.merge((y_ecualizado, cr, cb))
    imagen_ecualizada = cv2.cvtColor(imagen_ecualizada, cv2.COLOR_YCrCb2RGB)

    # Convertimos el array de NumPy a una imagen Pillow
    resultado = Image.fromarray(imagen_ecualizada)

    # Creamos una figura para comparar ambas imágenes
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5))

    ejes[0].imshow(imagen)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(resultado)
    ejes[1].set_title("Imagen ecualizada")
    ejes[1].axis("off")

    figura.tight_layout()

    # Guardamos la figura si se indicó una ruta
    if guardar_en:
        figura.savefig(guardar_en, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figura)
    return resultado


# ----------------- (3) -----------------
# Función para aplicar filtros de Pillow
def aplicar_filtro(imagen, nombre_filtro, guardar_en=None):
    """Aplica uno de los nueve filtros disponibles de Pillow y muestra
    una comparación entre la imagen original y la imagen filtrada."""
    # Diccionario con los filtros disponibles
    filtros = {
        "blur": ImageFilter.BLUR,
        "contour": ImageFilter.CONTOUR,
        "detail": ImageFilter.DETAIL,
        "edge enhance": ImageFilter.EDGE_ENHANCE,
        "edge enhance more": ImageFilter.EDGE_ENHANCE_MORE,
        "emboss": ImageFilter.EMBOSS,
        "find edges": ImageFilter.FIND_EDGES,
        "sharpen": ImageFilter.SHARPEN,
        "smooth": ImageFilter.SMOOTH}

    # Normalizamos el nombre ingresado
    nombre = nombre_filtro.lower().strip()

    # Verificamos que el filtro exista
    if nombre not in filtros:
        raise ValueError("Nombre de filtro no válido")

    # Aplicamos el filtro seleccionado
    resultado = imagen.filter(filtros[nombre])

    # Creamos una figura para comparar ambas imágenes
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5))

    ejes[0].imshow(imagen)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(resultado)
    ejes[1].set_title(nombre.title())
    ejes[1].axis("off")

    figura.tight_layout()

    # Guardamos la figura si se indicó una ruta
    if guardar_en:
        figura.savefig(guardar_en, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figura)
    return resultado


# ----------------- (4) -----------------
# Función para detectar bordes y contornos
def detectar_bordes(imagen, tecnica, guardar_en=None):
    """Detecta bordes y contornos de una imagen utilizando la técnica
    Sobel o Canny. Muestra y guarda una comparación entre la imagen
    original y el resultado obtenido."""
    # Convertimos la imagen a un array de NumPy
    imagen_array = np.array(imagen)

    # Convertimos la imagen a escala de grises
    gris = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2GRAY)

    # Normalizamos el nombre ingresado
    tecnica = tecnica.lower().strip()

    # Aplicamos la técnica seleccionada
    if tecnica == "sobel":
        gradiente_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        gradiente_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        magnitud = np.sqrt(gradiente_x**2 + gradiente_y**2)
        bordes = cv2.convertScaleAbs(magnitud)
    
    elif tecnica == "canny":
        bordes = cv2.Canny(gris, 100, 200)
    else:
        raise ValueError("Técnica no válida. Elegí Sobel o Canny.")

    # Convertimos el resultado a una imagen Pillow
    resultado = Image.fromarray(bordes)

    # Mostramos la imagen original y el resultado
    figura, ejes = plt.subplots(1, 2, figsize=(10, 5))

    ejes[0].imshow(imagen)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(resultado, cmap="gray")
    ejes[1].set_title(f"Detección de bordes: {tecnica.title()}")
    ejes[1].axis("off")

    figura.tight_layout()

    # Guardamos la comparación si se indicó una ruta
    if guardar_en:
        figura.savefig(guardar_en, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(figura)
    return resultado

# ----------------- (5) -----------------
# Menú principal de la aplicación
def main(carpeta_resultados):
    """Ejecuta el menú principal de FOTOAPP y permite al usuario
    cargar imágenes, redimensionarlas, ajustar su contraste,
    aplicar filtros y detectar bordes."""
    imagen_actual = None
    while True:
        print("\n========== FOTOAPP ==========")
        print("1. Cargar y redimensionar imagen")
        print("2. Ajustar contraste")
        print("3. Aplicar filtro")
        print("4. Detectar bordes y contornos")
        print("5. Salir")
        opcion = input("\nSeleccione una opción: ").strip()

        # Opción 1: cargar imagen
        if opcion == "1":
            ruta = input("Ingrese la ruta de la imagen: ")
            plataforma = input("Ingrese la plataforma (Youtube, Instagram, Twitter o Facebook): ")
            try:
                nombre_archivo = f"redimensionada_{plataforma.lower().strip()}.png"
                ruta_guardado = Path(carpeta_resultados) / nombre_archivo
                imagen_actual = leer_y_redimensionar(ruta, plataforma, ruta_guardado)
                print("Imagen cargada, redimensionada y guardada correctamente.")
            except (FileNotFoundError, ValueError, OSError) as error:
                print(f"Error: {error}")

        # Verificamos que exista una imagen cargada
        elif opcion in ["2", "3", "4"]:
            if imagen_actual is None:
                print("Primero debe cargar una imagen utilizando la opción 1.")
                continue
            try:
                # Opción 2: ajustar contraste
                if opcion == "2":
                    ruta_guardado = Path(carpeta_resultados) / "contraste_menu.png"
                    ajustar_contraste_histograma(imagen_actual, ruta_guardado)

                # Opción 3: aplicar filtro
                elif opcion == "3":
                    filtro = input(
                        "Ingresá el filtro (Blur, Contour, Detail, "
                        "Edge Enhance, Edge Enhance More, Emboss, "
                        "Find Edges, Sharpen, Smooth): ")
                    nombre_archivo = filtro.lower().replace(" ", "_")
                    ruta_guardado = Path(carpeta_resultados) / f"filtro_{nombre_archivo}.png"
                    aplicar_filtro(imagen_actual, filtro, ruta_guardado)

                # Opción 4: detectar bordes
                elif opcion == "4":
                    tecnica = input("Ingresá la técnica (Sobel o Canny): ")
                    nombre_archivo = tecnica.lower().strip()
                    ruta_guardado = Path(carpeta_resultados) / f"bordes_{nombre_archivo}.png"
                    detectar_bordes(imagen_actual, tecnica, ruta_guardado)
            
            except (ValueError, TypeError, OSError) as error:
                print(f"Error: {error}")

        # Opción 5: salir
        elif opcion == "5":
            print("Gracias por utilizar FOTOAPP. ¡Hasta luego!")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

# Ejecutamos el menú si se inicia este archivo directamente
if __name__ == "__main__":
    main("Resultados")
    