from customtkinter import *

# Definimos los basicos del frame
HEIGTH, WIDTH = 600, 500
app = CTk()
app.geometry(f'{HEIGTH}x{WIDTH}')
app.title('Depression Predicter')

# Variables basicas de la logica
respuestas: dict = {
    'gender': 0,
    'age': 'Age',
    'study hours': 'Study Hours',
    'academic pressure': 0,
    'financial stress': 0,
    'study satisfaction': 0,
    'sleep duration': 0,
    'dietary habits': 0,
    'suicidal thouthts': False,
    'family background': False,
}

texto_resultado = StringVar()
texto_academic = StringVar()
texto_financial = StringVar()
texto_satisfaction = StringVar()

# Funciones de los widgets
def ready_answers(respuestas: dict) -> bool:
    for key in respuestas.keys():
        if respuestas[key] == 0:
            return False
    return 'normal'

def on_entry_age_change(value):
    respuestas['age'] = value

def on_checkbox_family_change(value):
    print(value)

def on_checkbox_suicidal_change(value):
    print(value)

def on_combo_change_gender(value):
    print(value)

def on_combo_change_diet(value):
    print(value)

def on_combo_change_sleep(value):
    print(value)

def on_slider_change_academic(value):
    texto_academic.set(int(value))

def on_slider_change_financial(value):
    texto_financial.set(int(value))

def on_slider_change_satisfaction(value):
    texto_satisfaction.set(int(value))

# Definimos los widgets
entry_age = CTkEntry(app, justify='center',
                     width=60, height=40,
                     border_color='#555555',
                     placeholder_text='Age', placeholder_text_color='#FFFFFF'
                     )
entry_study = CTkEntry(app, justify='center',
                     width=100, height=40,
                     border_color='#555555',
                     placeholder_text='Study Hours', placeholder_text_color='#FFFFFF'
                     )
checkbox_family_history = CTkCheckBox(app, text='Family depression history?',
                                      border_width=1, checkmark_color='#92ba41', fg_color='#555555',
                                      border_color='#92ba41', hover_color='#555555')
checkbox_suicidal_thoughts = CTkCheckBox(app, text='Has had suicidal thoughts?',
                                        border_width=1, checkmark_color='#92ba41', fg_color='#555555',
                                        border_color='#92ba41', hover_color='#555555')

combo_gender = CTkComboBox(app, values=['Male', 'Female'], justify='center',
                           width=175, height=40,
                           command=on_combo_change_gender,
                           border_color='#555555', button_color='#92ba41',
                           button_hover_color='#f59425', 
                           ); combo_gender.set('Gender')
combo_diet = CTkComboBox(app, values=['Healthy Diet', 'Moderate Diet', 'Unhealthy Diet'], justify='center',
                           width=175, height=40,
                           command=on_combo_change_diet,
                           border_color='#555555', button_color='#92ba41',
                           button_hover_color='#f59425', 
                           ); combo_diet.set('Dietary Habits')
combo_sleep = CTkComboBox(app, values=['More than 8 hours', '7-8 hours', '5-6 hours', 'Less than 5 hours'], justify='center',
                           width=175, height=40,
                           command=on_combo_change_sleep,
                           border_color='#555555', button_color='#92ba41',
                           button_hover_color='#f59425', 
                           ); combo_sleep.set('Sleep')

frame_academic = CTkFrame(app, width=WIDTH * 1.1, height=HEIGTH * 0.0625, 
                            border_color='#555555', border_width=1
                            )
label_academic = CTkLabel(frame_academic, text='Academic Pressure')
slider_academic = CTkSlider(
    master=frame_academic,
    width=300, height=15,
    button_color='#92ba41', button_hover_color='#f59425',
    from_=1, to=5, number_of_steps=4,
    command=on_slider_change_academic,
)
label_academic_res = CTkLabel(
    frame_academic,
    textvariable=texto_academic,
    font=('arial', 15)
)

frame_financial = CTkFrame(app, width=WIDTH * 1.1, height=HEIGTH * 0.0625, 
                            border_color='#555555', border_width=1
                            )
label_financial = CTkLabel(frame_financial, text='Financial Pressure')
slider_financial = CTkSlider(
    master=frame_financial,
    width=300, height=15,
    button_color='#92ba41', button_hover_color='#f59425',
    from_=1, to=5, number_of_steps=4,
    command=on_slider_change_financial,
)
label_financial_res = CTkLabel(
    frame_financial,
    textvariable=texto_financial,
    font=('arial', 15)
)

frame_satisfaction = CTkFrame(app, width=WIDTH * 1.1, height=HEIGTH * 0.0625, 
                            border_color='#555555', border_width=1
                            )
label_satisfaction = CTkLabel(frame_satisfaction, text='Academic Satisfaction')
slider_satisfaction = CTkSlider(
    master=frame_satisfaction,
    width=300, height=15,
    button_color='#92ba41', button_hover_color='#f59425',
    from_=1, to=5, number_of_steps=4,
    command=on_slider_change_satisfaction,
)
label_satisfaction_res = CTkLabel(
    frame_satisfaction,
    textvariable=texto_satisfaction,
    font=('arial', 15)
)

frame_resultados = CTkFrame(app, width=WIDTH * 1.1, height=HEIGTH * 0.15, 
                            border_color='#555555', border_width=1
                            )
bttn_calcular = CTkButton(frame_resultados, text='Calculate Results', state=ready_answers(respuestas),
                          height=40, width=150,
                          text_color='#000000', fg_color='#92ba41', hover_color='#f59425',
                          border_color='#555555', border_width=1,
                          )
label_resultado = CTkLabel(frame_resultados, text=texto_resultado)


# Desplegamos los widgets
entry_age.place(x=25, y=HEIGTH * 0.05)
entry_study.place(x=95, y=HEIGTH * 0.05)

combo_gender.place(x=25, y= HEIGTH * 0.15)
combo_diet.place(x=212.5, y= HEIGTH * 0.15)
combo_sleep.place(x=400, y= HEIGTH * 0.15)

frame_satisfaction.place(x= WIDTH * 0.05, y= HEIGTH * 0.25)
label_satisfaction.place(x=25, y= 4)
slider_satisfaction.place(x=175, y=11)
label_satisfaction_res.place(x=510, y= 4)

frame_academic.place(x= WIDTH * 0.05, y= HEIGTH * 0.35)
label_academic.place(x=25, y= 4)
slider_academic.place(x=175, y=11)
label_academic_res.place(x=510, y= 4)

frame_financial.place(x= WIDTH * 0.05, y= HEIGTH * 0.45)
label_financial.place(x=25, y= 4)
slider_financial.place(x=175, y=11)
label_financial_res.place(x=510, y= 4)

checkbox_family_history.place(x=25, y=HEIGTH * 0.55)
checkbox_suicidal_thoughts.place(x=275, y=HEIGTH * 0.55)

frame_resultados.place(x= WIDTH * 0.05, y= HEIGTH * 0.65)
bttn_calcular.place(x=25, y=25)
label_resultado.place(x = 300, y = 30)

# Lanzamos la interfaz
app.mainloop()