# --- src/gui/strategy_selector_window.py ---

import customtkinter as ctk
from tkinter import messagebox
from typing import Dict, Optional, Type

from strategies.base_strategy import BettingStrategy
from strategies.draw_strategy import BackDrawStrategy
from strategies.lay_away_strategy import LayAwayStrategy

class StrategySelectorWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Seleção de Estratégia")
        self.geometry("450x200")
        self.resizable(False, False)
        self.eval('tk::PlaceWindow . center')

        self.available_strategies: Dict[str, Type[BettingStrategy]] = {
            "DrawEnsemble (ML)": BackDrawStrategy,
            "LayAway (Heuristic)": LayAwayStrategy,
        }
        
        self.selected_strategy_class: Optional[Type[BettingStrategy]] = None
        self.strategy_var = ctk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        """Cria e posiciona os widgets na janela de seleção."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)

        info_label = ctk.CTkLabel(
            main_frame, 
            text="Escolha o mercado que deseja analisar e prever:",
            font=ctk.CTkFont(size=14)
        )
        info_label.pack(pady=(0, 15), anchor="w")

        strategy_names = list(self.available_strategies.keys())
        self.strategy_combo = ctk.CTkComboBox(
            main_frame, 
            variable=self.strategy_var, 
            values=strategy_names, 
            state="readonly",
            font=ctk.CTkFont(size=12)
        )
        if strategy_names:
            self.strategy_combo.set(strategy_names[0])
        self.strategy_combo.pack(fill="x", ipady=4) 

        start_button = ctk.CTkButton(
            main_frame, 
            text="Iniciar Aplicação", 
            command=self.start_application,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        start_button.pack(pady=(20, 0), anchor="e")

    def start_application(self):
        """
        Callback para o botão "Iniciar". Pega a estratégia selecionada,
        armazena a classe correspondente e fecha esta janela.
        """
        selected_name = self.strategy_var.get()
        if not selected_name:
            messagebox.showwarning(
                "Nenhuma Estratégia", 
                "Por favor, selecione uma estratégia para continuar.", 
                parent=self
            )
            return

        self.selected_strategy_class = self.available_strategies[selected_name]
        
        self.destroy()

    def get_selected_strategy_class(self) -> Optional[Type[BettingStrategy]]:
        """
        Método público para o app_launcher obter a classe da estratégia que foi escolhida.
        Retorna a classe em si, não uma instância.
        """
        return self.selected_strategy_class