from customtkinter import *

# Gender, Age, Academic Pressure, Study Satisfaction, Sleep Duration, Dietary Habits, 
# Have you ever had suicidal thoughts?, Work/Study Hours, Financial Stress, Family History of Mental Illness,

# Variables basicas de la logica
respuestas: dict = {
    'gender': 0,
    'age': 0,
    'academic pressure': 0,
    'study satisfaction': 0,
    'sleep duration': 0,
    'dietary habits': 0,
    'suicidal thouthts': 0,
    'financial stress': 0,
    'family background': 0,
}

# Funciones del boton
def ready_answers(respuestas: dict) -> bool: # Devuelve True cuando todas las respuestas sean distintas de 0
    for key in respuestas.keys():
        if respuestas[key] == 0:
            return False
    return 'normal'

# Definimos los basicos del frame
HEIGTH, WIDTH = 600, 500
app = CTk()
app.geometry(f'{HEIGTH}x{WIDTH}')
app.title('Depression Predicter')

# Definimos los widgets
frame_resultados = CTkFrame(app, width=WIDTH * 1.1, height=HEIGTH * 0.15, 
                            border_color='#333333', border_width=1
                            )

bttn_calcular = CTkButton(frame_resultados, text='Calcular Resultado', state=ready_answers(respuestas),
                          height=40, width=150,
                          text_color='#000000', fg_color='#92ba41', hover_color='#f59425',
                          border_color='#333333', border_width=1,
                          )


# Desplegamos los widgets
frame_resultados.place(x= WIDTH * 0.05, y= HEIGTH * 0.65)
bttn_calcular.place(x=25, y=25)

# Lanzamos la interfaz
app.mainloop()