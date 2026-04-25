# Developed by ODAT project
# please see https://odat.info
# please see https://github.com/ODAT-Project

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb 
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy.stats import skew
from sklearn.metrics import confusion_matrix
import gc 

STYLE_CONFIG = {
    "font_family": "Segoe UI",
    "font_size_normal": 10,
    "font_size_header": 13,
    "font_size_section": 11,
    "bg_root": "#F5F6FA",        
    "bg_widget": "#FFFFFF",      
    "bg_entry": "#F8F9FA",       
    "fg_text": "#2F3640",        
    "fg_header": "#192A56",      
    "accent_color": "#0097E6",   
    "accent_text_color": "#FFFFFF",
    "border_color": "#DCDDE1",
    "listbox_select_bg": "#0097E6",
    "listbox_select_fg": "#FFFFFF",
    "disabled_bg": "#EAECEF",
    "disabled_fg": "#A4B0BE",
    "error_text_color": "#E84118",
}

class DynamicCVDApp:
    XGBOOST_RISK_SCORE_COL_NAME = 'XGBoost_Risk_Score_Covariate'

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Dynamic CVD Risk Predictor -- XGBoost-CPH Enhanced")
        self.root.geometry("1450x980")
        self.root.configure(bg=STYLE_CONFIG["bg_root"])

        self.data_df = None
        self.xgb_survival_model = None 
        self.cph_hybrid_model = None          
        self.cph_standard_model = None
        
        self.scaler_xgb = None         
        self.scaler_cph_linear = None  
        self.num_imputer_xgb = None
        self.num_imputer_cph_linear = None

        self.trained_xgb_survival_feature_names = []
        self.trained_cph_linear_feature_names = [] 
        self.all_base_features_for_input = [] 

        self.trained_feature_medians_xgb = {}
        self.trained_feature_medians_cph_linear = {}
        self.scaled_columns_xgb = []
        self.scaled_columns_cph_linear = []

        self.target_event_col_var = tk.StringVar()
        self.time_horizon_var = tk.StringVar(value="5") 
        self.time_to_event_col_var = tk.StringVar() 

        self.n_estimators_var = tk.StringVar(value="100")
        self.max_depth_var = tk.StringVar(value="3")
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.cph_penalizer_var = tk.StringVar(value="0.1") 

        self.y_test_full_df_for_metrics = None
        self.test_df_raw = None
        self.prediction_input_widgets = {}
        self.dynamic_input_scrollable_frame = None
        self.more_plots_window = None
        self.current_report_lines = []
        self.metrics_dict = {}

        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    def setup_styles(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        
        font_normal = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"])
        font_bold = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"], "bold")
        font_header = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_header"], "bold")
        font_section = (STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_section"], "bold")

        s.configure(".", font=font_normal, background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        s.configure("TFrame", background=STYLE_CONFIG["bg_root"])
        s.configure("Content.TFrame", background=STYLE_CONFIG["bg_widget"])
        s.configure("TLabel", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_text"])
        s.configure("Header.TLabel", font=font_header, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_root"]) 
        s.configure("Section.TLabel", font=font_section, foreground=STYLE_CONFIG["fg_header"], background=STYLE_CONFIG["bg_widget"]) 
        
        s.configure("TButton", font=font_bold, padding=6, background=STYLE_CONFIG["accent_color"], foreground=STYLE_CONFIG["accent_text_color"], borderwidth=0)
        s.map("TButton", 
              background=[("active", "#0086CC"), ("disabled", STYLE_CONFIG["disabled_bg"])], 
              foreground=[("active", STYLE_CONFIG["accent_text_color"]), ("disabled", STYLE_CONFIG["disabled_fg"])])
        
        s.configure("TEntry", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], bordercolor=STYLE_CONFIG["border_color"])
        s.configure("TCombobox", fieldbackground=STYLE_CONFIG["bg_entry"], foreground=STYLE_CONFIG["fg_text"], bordercolor=STYLE_CONFIG["border_color"])
        
        self.root.option_add('*TCombobox*Listbox.background', STYLE_CONFIG["bg_entry"])
        self.root.option_add('*TCombobox*Listbox.foreground', STYLE_CONFIG["fg_text"])
        self.root.option_add('*TCombobox*Listbox.selectBackground', STYLE_CONFIG["listbox_select_bg"])
        self.root.option_add('*TCombobox*Listbox.selectForeground', STYLE_CONFIG["listbox_select_fg"])

        s.configure("TLabelFrame", background=STYLE_CONFIG["bg_widget"], bordercolor=STYLE_CONFIG["border_color"])
        s.configure("TLabelFrame.Label", background=STYLE_CONFIG["bg_widget"], foreground=STYLE_CONFIG["fg_header"], font=font_section)

    def create_menu(self):
        menubar = tk.Menu(self.root, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"])
        file_menu = tk.Menu(menubar, tearoff=0, bg=STYLE_CONFIG["bg_widget"], fg=STYLE_CONFIG["fg_text"])
        file_menu.add_command(label="Load CSV...", command=self.load_csv_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="About", command=self.show_about_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.bind_all("<Control-o>", lambda e: self.load_csv_file())
        self.root.bind_all("<Control-q>", lambda e: self.root.quit())

    def create_main_layout(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        train_config_pane = ttk.Frame(main_pane, padding="10", style="Content.TFrame")
        main_pane.add(train_config_pane, weight=1) 

        predict_results_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(predict_results_pane, weight=1) 

        self.prediction_input_outer_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        predict_results_pane.add(self.prediction_input_outer_frame, weight=2)

        results_display_frame = ttk.Frame(predict_results_pane, padding="10", style="Content.TFrame")
        predict_results_pane.add(results_display_frame, weight=3)

        self.create_train_config_widgets(train_config_pane)
        self.create_dynamic_prediction_inputs_placeholder(self.prediction_input_outer_frame)
        self.create_results_display_widgets(results_display_frame)

    def log_training_message(self, message, is_error=False):
        if not hasattr(self, 'training_log_text') or not self.training_log_text.winfo_exists():
            return
        try:
            self.training_log_text.configure(state=tk.NORMAL)
            tag = "error_tag" if is_error else "normal_tag"
            self.training_log_text.tag_configure("error_tag", foreground=STYLE_CONFIG["error_text_color"])
            self.training_log_text.tag_configure("normal_tag", foreground=STYLE_CONFIG["fg_text"])
            self.training_log_text.insert(tk.END, message + "\n", tag)
            self.training_log_text.see(tk.END)
            self.training_log_text.configure(state=tk.DISABLED)
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def create_train_config_widgets(self, parent_frame):
        ttk.Label(parent_frame, text="Model Training Configuration", style="Header.TLabel").pack(pady=(0,10), anchor=tk.W)

        load_button = ttk.Button(parent_frame, text="Load CSV File", command=self.load_csv_file)
        load_button.pack(pady=5, fill=tk.X)
        
        self.loaded_file_label = ttk.Label(parent_frame, text="No file loaded.")
        self.loaded_file_label.pack(pady=(2,5), anchor=tk.W)

        target_config_frame = ttk.LabelFrame(parent_frame, text="Target Variable & Time")
        target_config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_config_frame, text="Event Column (1=event, 0=censor):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.target_event_selector = ttk.Combobox(target_config_frame, textvariable=self.target_event_col_var, state="readonly", width=28)
        self.target_event_selector.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(target_config_frame, text="Time to Event/Censor Column:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.time_to_event_selector = ttk.Combobox(target_config_frame, textvariable=self.time_to_event_col_var, state="readonly", width=28)
        self.time_to_event_selector.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)

        ttk.Label(target_config_frame, text="Prediction Horizon (Years):").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.time_horizon_entry = ttk.Entry(target_config_frame, textvariable=self.time_horizon_var, width=10)
        self.time_horizon_entry.grid(row=2, column=1, padx=5, pady=3, sticky=tk.W)
        target_config_frame.columnconfigure(1, weight=1)

        xgb_fs_frame = ttk.LabelFrame(parent_frame, text="Features for XGBoost Risk Score")
        xgb_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        xgb_listbox_container = ttk.Frame(xgb_fs_frame, style="Content.TFrame")
        xgb_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.xgb_feature_listbox = tk.Listbox(xgb_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=6,
                                              bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"],
                                              selectbackground=STYLE_CONFIG["listbox_select_bg"], 
                                              selectforeground=STYLE_CONFIG["listbox_select_fg"], relief=tk.FLAT, highlightthickness=1, highlightcolor=STYLE_CONFIG["border_color"])
        xgb_feature_listbox_scrollbar = ttk.Scrollbar(xgb_listbox_container, orient=tk.VERTICAL, command=self.xgb_feature_listbox.yview)
        self.xgb_feature_listbox.configure(yscrollcommand=xgb_feature_listbox_scrollbar.set)
        xgb_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.xgb_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        xgb_params_frame = ttk.LabelFrame(parent_frame, text="XGBoost Hyperparameters")
        xgb_params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(xgb_params_frame, text="Estimators:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.n_estimators_entry = ttk.Entry(xgb_params_frame, textvariable=self.n_estimators_var, width=8)
        self.n_estimators_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(xgb_params_frame, text="Max Depth:").grid(row=0, column=2, padx=5, pady=3, sticky=tk.W)
        self.max_depth_entry = ttk.Entry(xgb_params_frame, textvariable=self.max_depth_var, width=8)
        self.max_depth_entry.grid(row=0, column=3, padx=5, pady=3, sticky=tk.W)
        
        ttk.Label(xgb_params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.learning_rate_entry = ttk.Entry(xgb_params_frame, textvariable=self.learning_rate_var, width=8)
        self.learning_rate_entry.grid(row=1, column=1, padx=5, pady=3, sticky=tk.W)

        cph_fs_frame = ttk.LabelFrame(parent_frame, text="Features for Linear Part of CPH Model")
        cph_fs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        cph_listbox_container = ttk.Frame(cph_fs_frame, style="Content.TFrame")
        cph_listbox_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cph_linear_feature_listbox = tk.Listbox(cph_listbox_container, selectmode=tk.MULTIPLE, exportselection=False, height=5,
                                                     bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"],
                                                     selectbackground=STYLE_CONFIG["listbox_select_bg"], 
                                                     selectforeground=STYLE_CONFIG["listbox_select_fg"], relief=tk.FLAT, highlightthickness=1, highlightcolor=STYLE_CONFIG["border_color"])
        cph_linear_feature_listbox_scrollbar = ttk.Scrollbar(cph_listbox_container, orient=tk.VERTICAL, command=self.cph_linear_feature_listbox.yview)
        self.cph_linear_feature_listbox.configure(yscrollcommand=cph_linear_feature_listbox_scrollbar.set)
        cph_linear_feature_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cph_linear_feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cph_params_frame = ttk.LabelFrame(parent_frame, text="CPH Hyperparameters")
        cph_params_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cph_params_frame, text="L2 Penalizer:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.cph_penalizer_entry = ttk.Entry(cph_params_frame, textvariable=self.cph_penalizer_var, width=8)
        self.cph_penalizer_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.W)

        self.train_button = ttk.Button(parent_frame, text="Train Hybrid XGBoost-CPH Model", command=self.train_model_action)
        self.train_button.pack(pady=(10,5), fill=tk.X)

        ttk.Label(parent_frame, text="Training Log & Report:", style="Section.TLabel").pack(anchor=tk.W, pady=(10,0))
        self.training_log_text = scrolledtext.ScrolledText(parent_frame, height=5, wrap=tk.WORD, bg=STYLE_CONFIG["bg_entry"], fg=STYLE_CONFIG["fg_text"], relief=tk.FLAT, highlightthickness=1, highlightbackground=STYLE_CONFIG["border_color"])
        self.training_log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.training_log_text.configure(state=tk.DISABLED)

    def create_dynamic_prediction_inputs_placeholder(self, parent_frame):
        ttk.Label(parent_frame, text="Patient Data for Prediction", style="Header.TLabel").pack(pady=(0,10), anchor=tk.W)
        self.dynamic_input_canvas = tk.Canvas(parent_frame, borderwidth=0, background=STYLE_CONFIG["bg_widget"], highlightthickness=0)
        vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=self.dynamic_input_canvas.yview)
        self.dynamic_input_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.dynamic_input_canvas.pack(side="left", fill="both", expand=True)
        
        self.dynamic_input_scrollable_frame = ttk.Frame(self.dynamic_input_canvas, style="Content.TFrame")
        self.dynamic_input_canvas.create_window((0, 0), window=self.dynamic_input_scrollable_frame, anchor="nw")
        self.dynamic_input_scrollable_frame.bind("<Configure>", lambda e: self.dynamic_input_canvas.configure(scrollregion=self.dynamic_input_canvas.bbox("all")))
        
        self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.")
        self.placeholder_pred_label.pack(padx=10, pady=20)
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (Hybrid CPH)", command=self.assess_risk_action)
        self.assess_button.pack_forget()

    def create_dynamic_prediction_inputs(self):
        if self.dynamic_input_scrollable_frame:
            for widget in self.dynamic_input_scrollable_frame.winfo_children():
                widget.destroy()
        self.prediction_input_widgets = {}

        self.all_base_features_for_input = sorted(list(set(self.trained_xgb_survival_feature_names + self.trained_cph_linear_feature_names)))

        if not self.all_base_features_for_input:
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="No features available. Train model.")
            self.placeholder_pred_label.pack(padx=10, pady=20)
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()
            return

        for feature_name in self.all_base_features_for_input:
            row_frame = ttk.Frame(self.dynamic_input_scrollable_frame, style="Content.TFrame")
            row_frame.pack(fill=tk.X, pady=1, padx=2)
            
            display_name = feature_name if len(feature_name) < 35 else feature_name[:32] + "..."
            label = ttk.Label(row_frame, text=f"{display_name}:", width=35, anchor="w")
            label.pack(side=tk.LEFT, padx=(0,5))
            
            entry = ttk.Entry(row_frame, font=(STYLE_CONFIG["font_family"], STYLE_CONFIG["font_size_normal"]))
            default_val = "0"
            if feature_name in self.trained_feature_medians_xgb:
                 default_val = self.trained_feature_medians_xgb.get(feature_name, "0")
            elif feature_name in self.trained_feature_medians_cph_linear:
                 default_val = self.trained_feature_medians_cph_linear.get(feature_name, "0")

            entry.insert(0, str(default_val))
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.prediction_input_widgets[feature_name] = entry

        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (Hybrid CPH)", command=self.assess_risk_action)
        self.assess_button.pack(pady=(15,10), fill=tk.X, padx=5)
        self.dynamic_input_scrollable_frame.update_idletasks()
        self.dynamic_input_canvas.config(scrollregion=self.dynamic_input_canvas.bbox("all"))

    def create_results_display_widgets(self, parent_frame):
        top_frame = ttk.Frame(parent_frame, style="Content.TFrame")
        top_frame.pack(fill=tk.X, pady=5)

        pred_res_frame = ttk.LabelFrame(top_frame, text="Hybrid Prediction Result")
        pred_res_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.risk_prob_label = ttk.Label(pred_res_frame, text="N/A", font=(STYLE_CONFIG["font_family"], 22, "bold"), foreground=STYLE_CONFIG["accent_color"])
        self.risk_prob_label.pack(pady=(5,2))
        self.risk_interpretation_label = ttk.Label(pred_res_frame, text="Train model & assess.")
        self.risk_interpretation_label.pack(pady=(0,5))

        btn_container = ttk.Frame(top_frame, style="Content.TFrame")
        btn_container.pack(side=tk.RIGHT, padx=(5,0), pady=10, anchor="ne")

        self.save_results_button = ttk.Button(btn_container, text="Export All Results to Folder", command=self.save_results_action, state=tk.DISABLED)
        self.save_results_button.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        self.more_plots_button = ttk.Button(btn_container, text="View Survival Plots", command=self.show_more_plots_window, state=tk.DISABLED)
        self.more_plots_button.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.LabelFrame(parent_frame, text="Model Performance Visuals")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        plt.style.use('default')
        self.fig = plt.Figure(figsize=(8, 7), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        self.fig.subplots_adjust(hspace=0.5, wspace=0.3) 

        self.ax_importance_xgb = self.fig.add_subplot(2, 2, 1) 
        self.ax_cph_coeffs = self.fig.add_subplot(2, 2, 2)    
        self.ax_calibration_like = self.fig.add_subplot(2, 2, 3) 
        self.ax_survival_curve = self.fig.add_subplot(2, 2, 4) 

        for ax in [self.ax_importance_xgb, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]:
            ax.tick_params(colors=STYLE_CONFIG["fg_text"])
            ax.xaxis.label.set_color(STYLE_CONFIG["fg_text"])
            ax.yaxis.label.set_color(STYLE_CONFIG["fg_text"])
            ax.title.set_color(STYLE_CONFIG["fg_header"])
            ax.set_facecolor(STYLE_CONFIG["bg_entry"])
            for spine in ax.spines.values():
                spine.set_color(STYLE_CONFIG["border_color"])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=STYLE_CONFIG["bg_widget"])
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.update_plots(clear_only=True)

    def toggle_train_predict_sections_enabled(self, data_loaded=False, model_trained=False):
        train_state = tk.NORMAL if data_loaded else tk.DISABLED
        predict_state = tk.NORMAL if model_trained else tk.DISABLED

        for btn in ['train_button', 'xgb_feature_listbox', 'cph_linear_feature_listbox']:
            if hasattr(self, btn): getattr(self, btn).config(state=train_state)

        for btn in ['target_event_selector', 'time_to_event_selector']:
            if hasattr(self, btn): getattr(self, btn).config(state="readonly" if data_loaded else tk.DISABLED)

        for btn in ['time_horizon_entry', 'n_estimators_entry', 'max_depth_entry', 'learning_rate_entry', 'cph_penalizer_entry']:
            if hasattr(self, btn): getattr(self, btn).config(state=tk.NORMAL if data_loaded else tk.DISABLED)

        for btn in ['assess_button', 'more_plots_button', 'save_results_button']:
            if hasattr(self, btn) and getattr(self, btn).winfo_exists():
                getattr(self, btn).config(state=predict_state)

        for _feature_name, widget in self.prediction_input_widgets.items():
            if hasattr(widget, 'config'): widget.config(state=tk.NORMAL if model_trained else tk.DISABLED)

    def downcast_numerics(self, df):
        self.log_training_message("  Attempting to downcast numeric types for memory optimization...")
        f_cols = df.select_dtypes('float').columns
        i_cols = df.select_dtypes('integer').columns
        df[f_cols] = df[f_cols].apply(pd.to_numeric, downcast='float')
        df[i_cols] = df[i_cols].apply(pd.to_numeric, downcast='integer')
        gc.collect()
        return df

    def is_binary(self, series):
        return series.dropna().nunique() <= 2

    def _populate_ui_lists_after_load(self, column_names):
        self.xgb_feature_listbox.delete(0, tk.END)
        self.cph_linear_feature_listbox.delete(0, tk.END)
        for col_name in column_names:
            self.xgb_feature_listbox.insert(tk.END, col_name)
            self.cph_linear_feature_listbox.insert(tk.END, col_name)

        self.target_event_selector['values'] = column_names
        self.time_to_event_selector['values'] = column_names

        if column_names: self.target_event_col_var.set(column_names[0])
        if column_names and len(column_names) > 1: self.time_to_event_col_var.set(column_names[1])

        self.log_training_message(f"  UI lists populated with {len(column_names)} columns.")
        self.root.update_idletasks()

    def load_csv_file(self):
        filepath = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath: return
        try:
            self.data_df = pd.read_csv(filepath, low_memory=False)
            self.data_df = self.downcast_numerics(self.data_df.copy())
            gc.collect() 
            self.loaded_file_label.config(text=f"Loaded: {filepath.split('/')[-1]} ({self.data_df.shape[0]} rows, {self.data_df.shape[1]} cols)")
            self.log_training_message(f"Successfully loaded and downcasted {filepath.split('/')[-1]}.")
            
            column_names = sorted([str(col) for col in self.data_df.columns if str(col).strip()])
            if not column_names:
                messagebox.showerror("CSV Error", "No columns detected in CSV.")
                return

            self.root.after(10, self._populate_ui_lists_after_load, column_names)
            self.xgb_survival_model = None; self.cph_hybrid_model = None; self.cph_standard_model = None
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=False)
            self.update_plots(clear_only=True)
        except Exception as e:
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file.\nError: {e}")
            self.data_df = None
            self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)

    def _preprocess_features(self, df_subset, feature_names, stored_medians, scaler_to_use, scaled_cols_list, imputer_to_use, fit_transform=True):
        processed_df = df_subset[feature_names].copy()
        for col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        if fit_transform:
            for col in processed_df.columns: stored_medians[col] = processed_df[col].median()
            imputer_to_use.fit(processed_df)

        processed_df.loc[:,:] = imputer_to_use.transform(processed_df)

        if fit_transform:
            scaled_cols_list.clear()
            for col in processed_df.columns:
                if not pd.api.types.is_float_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].astype(np.float64)
                if processed_df[col].nunique(dropna=False) > 2:
                    scaled_cols_list.append(col)

        if scaled_cols_list:
            for col_to_scale in scaled_cols_list:
                if not pd.api.types.is_float_dtype(processed_df[col_to_scale]):
                    processed_df[col_to_scale] = processed_df[col_to_scale].astype(np.float64)
            if fit_transform:
                scaler_to_use.fit(processed_df[scaled_cols_list])
            if not processed_df[scaled_cols_list].empty:
                 scaled_values = scaler_to_use.transform(processed_df[scaled_cols_list])
                 processed_df.loc[:, scaled_cols_list] = scaled_values.astype(np.float32)
        return processed_df

    def evaluate_metrics(self, time_array, event_array, risk_scores, horizon_years):
        horizon_days = horizon_years * 365.25
        
        c_index = concordance_index(time_array, -risk_scores, event_array)

        valid_mask = ((event_array == 1) & (time_array <= horizon_days)) | (time_array > horizon_days)
        
        if not valid_mask.any():
            return {'C-index': c_index, 'Sens': np.nan, 'Spec': np.nan, 'PPV': np.nan, 'NPV': np.nan}
            
        y_true = np.where(time_array[valid_mask] <= horizon_days, 1, 0)
        valid_risks = risk_scores[valid_mask]
        
        cutoff = np.median(valid_risks)
        y_pred = np.where(valid_risks >= cutoff, 1, 0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        return {'C-index': c_index, 'Sens': sens, 'Spec': spec, 'PPV': ppv, 'NPV': npv}

    def train_model_action(self):
        if self.data_df is None: return

        selected_xgb_indices = self.xgb_feature_listbox.curselection()
        selected_cph_linear_indices = self.cph_linear_feature_listbox.curselection()
        target_event_col = self.target_event_col_var.get()
        time_to_event_col = self.time_to_event_col_var.get()

        if not selected_xgb_indices:
            messagebox.showerror("Error", "No features selected for XGBoost Risk Score.")
            return

        self.trained_xgb_survival_feature_names = [self.xgb_feature_listbox.get(i) for i in selected_xgb_indices]
        self.trained_cph_linear_feature_names = [self.cph_linear_feature_listbox.get(i) for i in selected_cph_linear_indices]
        self.all_base_features_for_input = sorted(list(set(self.trained_xgb_survival_feature_names + self.trained_cph_linear_feature_names)))

        try:
            n_estimators = int(self.n_estimators_var.get())
            max_depth = int(self.max_depth_var.get())
            learning_rate = float(self.learning_rate_var.get())
            cph_penalizer = float(self.cph_penalizer_var.get())
            pred_horizon = float(self.time_horizon_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid hyperparameter values.")
            return

        self.log_training_message("--- Starting Model Training ---")
        full_df_processed = self.data_df.copy()

        full_df_processed.dropna(subset=[target_event_col, time_to_event_col], inplace=True)
        full_df_processed = full_df_processed[full_df_processed[time_to_event_col] > 0]

        train_df, test_df = train_test_split(
            full_df_processed, test_size=0.25, random_state=42,
            stratify=full_df_processed[target_event_col] if full_df_processed[target_event_col].nunique() > 1 else None
        )
        self.test_df_raw = test_df.copy()

        # ---------------------------------------------------------
        #standard conventional Cox model
        # ---------------------------------------------------------
        self.log_training_message("\n--- Training Standard Conventional Cox Model (Baseline) ---")
        try:
            df_std_train = train_df[self.all_base_features_for_input + [time_to_event_col, target_event_col]].copy()
            df_std_train.dropna(inplace=True)
            self.cph_standard_model = CoxPHFitter(penalizer=cph_penalizer)
            self.cph_standard_model.fit(df_std_train, duration_col=time_to_event_col, event_col=target_event_col)
            self.log_training_message("  Standard Cox baseline model trained.")
        except Exception as e:
            self.log_training_message(f"Standard Cox failed (usually due to collinearity): {e}", is_error=True)
            self.cph_standard_model = None

        # ---------------------------------------------------------
        #XGBoost model
        # ---------------------------------------------------------
        self.log_training_message("\n--- Training XGBoost Survival Model ---")
        self.scaler_xgb = StandardScaler()
        self.num_imputer_xgb = SimpleImputer(strategy='median')
        self.trained_feature_medians_xgb = {}
        self.scaled_columns_xgb = []

        X_xgb_train = self._preprocess_features(train_df, self.trained_xgb_survival_feature_names, self.trained_feature_medians_xgb, self.scaler_xgb, self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=True)
        X_xgb_test = self._preprocess_features(test_df, self.trained_xgb_survival_feature_names, self.trained_feature_medians_xgb, self.scaler_xgb, self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=False)

        y_train_event = train_df[target_event_col].astype(int)
        y_train_time = train_df[time_to_event_col]
        y_xgb_survival_train = np.where(y_train_event == 1, y_train_time, -y_train_time)

        self.xgb_survival_model = xgb.XGBModel(objective='survival:cox', n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        self.xgb_survival_model.fit(X_xgb_train, y_xgb_survival_train)

        train_xgb_scores = self.xgb_survival_model.predict(X_xgb_train, output_margin=True)
        test_xgb_scores = self.xgb_survival_model.predict(X_xgb_test, output_margin=True)

        # ---------------------------------------------------------
        #hybrid CPH model
        # ---------------------------------------------------------
        self.log_training_message("\n--- Training Hybrid CPH Model ---")
        df_cph_train_fitting = pd.DataFrame(index=train_df.index)
        df_cph_test_fitting = pd.DataFrame(index=test_df.index)
        df_cph_train_fitting[DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME] = train_xgb_scores
        df_cph_test_fitting[DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME] = test_xgb_scores

        if self.trained_cph_linear_feature_names:
            self.scaler_cph_linear = StandardScaler()
            self.num_imputer_cph_linear = SimpleImputer(strategy='median')
            self.trained_feature_medians_cph_linear = {}
            self.scaled_columns_cph_linear = []

            X_cph_linear_train = self._preprocess_features(train_df, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=True)
            X_cph_linear_test = self._preprocess_features(test_df, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False)

            for col in X_cph_linear_train.columns:
                df_cph_train_fitting[col] = X_cph_linear_train[col]
                df_cph_test_fitting[col] = X_cph_linear_test[col]

        df_cph_train_fitting[time_to_event_col] = train_df[time_to_event_col].values
        df_cph_train_fitting[target_event_col] = train_df[target_event_col].astype(int).values
        df_cph_test_fitting[time_to_event_col] = test_df[time_to_event_col].values
        df_cph_test_fitting[target_event_col] = test_df[target_event_col].astype(int).values

        self.cph_hybrid_model = CoxPHFitter(penalizer=cph_penalizer, l1_ratio=0.0) 
        self.cph_hybrid_model.fit(df_cph_train_fitting, duration_col=time_to_event_col, event_col=target_event_col)
        self.y_test_full_df_for_metrics = df_cph_test_fitting.copy()

        # ---------------------------------------------------------
        #evaluate models on test set
        # ---------------------------------------------------------
        y_test_time = test_df[time_to_event_col].values
        y_test_event = test_df[target_event_col].astype(int).values
        
        #XGBoost Metrics
        self.metrics_dict['XGBoost'] = self.evaluate_metrics(y_test_time, y_test_event, test_xgb_scores, pred_horizon)

        #Hybrid CPH Metrics
        hybrid_risk_scores = self.cph_hybrid_model.predict_partial_hazard(df_cph_test_fitting).values
        self.metrics_dict['Hybrid_CPH'] = self.evaluate_metrics(y_test_time, y_test_event, hybrid_risk_scores, pred_horizon)

        #Standard Cox Metrics
        if self.cph_standard_model:
            df_std_test = test_df[self.all_base_features_for_input].copy()
            df_std_test.dropna(inplace=True)
            idx = df_std_test.index
            std_risk_scores = self.cph_standard_model.predict_partial_hazard(df_std_test).values
            self.metrics_dict['Standard_Cox'] = self.evaluate_metrics(
                test_df.loc[idx, time_to_event_col].values, 
                test_df.loc[idx, target_event_col].astype(int).values, 
                std_risk_scores, pred_horizon
            )
        else:
            self.metrics_dict['Standard_Cox'] = None

        self.generate_training_report(target_event_col, time_to_event_col, n_estimators, max_depth, learning_rate, cph_penalizer)
        self.create_dynamic_prediction_inputs() 
        self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=True)
        self.update_plots()

    def generate_training_report(self, target_col, time_col, n_est, max_d, lr, cph_pen):
        self.current_report_lines = [] 
        self.current_report_lines.append("\n=========================================")
        self.current_report_lines.append("--- COMBINED MODEL TRAINING REPORT ---")
        self.current_report_lines.append("=========================================")
        self.current_report_lines.append(f"Target Event Column: {target_col}")
        self.current_report_lines.append(f"Time Column: {time_col}")
        self.current_report_lines.append(f"Prediction Horizon: {self.time_horizon_var.get()} Years\n")

        self.current_report_lines.append("--- MODEL PERFORMANCE ON TEST SET ---")
        for mod_name, metrics in self.metrics_dict.items():
            if metrics:
                self.current_report_lines.append(f"\n{mod_name.replace('_', ' ')} Performance:")
                self.current_report_lines.append(f"  C-Index (Concordance): {metrics['C-index']:.4f}")
                self.current_report_lines.append(f"  Sensitivity:           {metrics['Sens']:.4f}")
                self.current_report_lines.append(f"  Specificity:           {metrics['Spec']:.4f}")
                self.current_report_lines.append(f"  PPV:                   {metrics['PPV']:.4f}")
                self.current_report_lines.append(f"  NPV:                   {metrics['NPV']:.4f}")
            else:
                self.current_report_lines.append(f"\n{mod_name} Performance: FAILED TO TRAIN")

        self.current_report_lines.append("\n=========================================")
        self.current_report_lines.append("--- HYBRID CPH MODEL DETAILS ---")
        if self.cph_hybrid_model:
            summary_df = self.cph_hybrid_model.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
            summary_df.rename(columns={'exp(coef)': 'Hazard Ratio (HR)', 'exp(coef) lower 95%': 'Lower 95% CI', 'exp(coef) upper 95%': 'Upper 95% CI'}, inplace=True)
            self.current_report_lines.extend(summary_df.to_string().split('\n'))

        self.current_report_lines.append("\n=========================================")
        self.current_report_lines.append("--- STANDARD COX BASELINE DETAILS ---")
        if self.cph_standard_model:
            std_summary = self.cph_standard_model.summary[['exp(coef)', 'p']]
            std_summary.rename(columns={'exp(coef)': 'Hazard Ratio (HR)'}, inplace=True)
            self.current_report_lines.extend(std_summary.to_string().split('\n'))
        else:
            self.current_report_lines.append("Model failed to converge (likely due to highly correlated/redundant features).")

        for line in self.current_report_lines:
            self.log_training_message(line, is_error=False)

    def save_results_action(self):
        if not self.current_report_lines: return
        
        folder_path = filedialog.askdirectory(title="Select Folder to Save All Results")
        if not folder_path: return
        
        try:
            #save report
            with open(os.path.join(folder_path, "01_Model_Training_Report.txt"), "w", encoding='utf-8') as f:
                for line in self.current_report_lines: f.write(line + "\n")
            
            #plots
            self.fig.savefig(os.path.join(folder_path, "03_Main_Dashboard_Plots.png"), dpi=300, bbox_inches='tight')

            #generate baseline table
            table_data = []
            for col in self.all_base_features_for_input:
                series = self.data_df[col]
                missing_pct = series.isnull().mean() * 100
                v_type = 'Binary' if self.is_binary(series) else 'Continuous'
                skewness = series.skew() if v_type == 'Continuous' else np.nan
                
                if v_type == 'Binary':
                    counts = series.value_counts()
                    val_1 = counts.get(1.0, counts.max()) 
                    info = f"{val_1} ({val_1/len(series.dropna())*100:.1f}%)"
                else:
                    info = f"{series.mean():.2f} ± {series.std():.2f}"
                
                table_data.append([col, v_type, f"{missing_pct:.1f}%", f"{skewness:.2f}", info])
            
            pd.DataFrame(table_data, columns=['Variable', 'Type', 'Missing (%)', 'Skewness', 'Baseline Info']).to_csv(
                os.path.join(folder_path, "02_Baseline_Characteristics.csv"), index=False
            )

            #generate EDA plots
            eda_dir = os.path.join(folder_path, "EDA_Plots")
            os.makedirs(eda_dir, exist_ok=True)

            self.log_training_message("\nGenerating EDA Plots to folder...")
            for col in self.all_base_features_for_input:
                fig_eda, ax_eda = plt.subplots(figsize=(6, 4), dpi=150)
                series = self.data_df[col].dropna()

                if self.is_binary(series):
                    sns.countplot(x=series, ax=ax_eda, palette="Blues")
                    ax_eda.set_title(f"Count Plot: {col}")
                else:
                    sns.histplot(series, ax=ax_eda, kde=True, color="#0097E6")
                    ax_eda.set_title(f"Histogram: {col}")

                fig_eda.tight_layout()

                safe_col_name = col.replace("/", "_").replace("\\", "_")

                fig_eda.savefig(os.path.join(eda_dir, f"{safe_col_name}_distribution.png"))
                plt.close(fig_eda)


            messagebox.showinfo("Success", f"All results, tables, and full-resolution plots saved to:\n\n{folder_path}")
            self.log_training_message("All saves completed successfully.")

        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")

    def assess_risk_action(self):
        if not self.xgb_survival_model or not self.cph_hybrid_model: return

        input_values_from_gui = {}
        try:
            for feature_name, widget in self.prediction_input_widgets.items():
                input_values_from_gui[feature_name] = float(widget.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical data.")
            return

        input_df_xgb = pd.DataFrame([input_values_from_gui])[self.trained_xgb_survival_feature_names]
        input_xgb_processed = self._preprocess_features(input_df_xgb, self.trained_xgb_survival_feature_names, self.trained_feature_medians_xgb, self.scaler_xgb, self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=False)
        new_xgb_risk_score = self.xgb_survival_model.predict(input_xgb_processed, output_margin=True)[0]

        cph_input_dict = {DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME: new_xgb_risk_score}
        
        if self.trained_cph_linear_feature_names:
            input_df_cph_linear = pd.DataFrame([input_values_from_gui])[self.trained_cph_linear_feature_names]
            input_cph_linear_processed = self._preprocess_features(input_df_cph_linear, self.trained_cph_linear_feature_names, self.trained_feature_medians_cph_linear, self.scaler_cph_linear, self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False)
            for col in input_cph_linear_processed.columns: cph_input_dict[col] = input_cph_linear_processed[col].iloc[0]

        input_data_for_cph_model = pd.DataFrame([cph_input_dict])[self.cph_hybrid_model.summary.index.tolist()]

        try:
            pred_horizon_years = float(self.time_horizon_var.get())
            survival_prob_at_horizon = self.cph_hybrid_model.predict_survival_function(input_data_for_cph_model, times=[pred_horizon_years * 365.25]).iloc[0,0]
            risk_at_horizon = 1.0 - survival_prob_at_horizon
            self.risk_prob_label.config(text=f"{pred_horizon_years:.0f}-Year Risk: {risk_at_horizon*100:.1f}%")

            if risk_at_horizon > 0.20: self.risk_interpretation_label.config(text="Higher Risk", foreground=STYLE_CONFIG["error_text_color"])
            elif risk_at_horizon > 0.10: self.risk_interpretation_label.config(text="Moderate Risk", foreground="#F39C12") 
            else: self.risk_interpretation_label.config(text="Lower Risk", foreground="#27AE60") 
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not make CPH prediction: {e}")

    def update_plots(self, clear_only=False):
        ax_list = [self.ax_importance_xgb, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]
        for ax in ax_list: ax.clear()

        title_font_dict = {'color': STYLE_CONFIG["fg_header"], 'fontsize': STYLE_CONFIG["font_size_normal"] + 1, 'weight': 'bold'}
        label_font_dict = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"] - 1}
        tick_color = STYLE_CONFIG["fg_text"]
        bar_color = STYLE_CONFIG["accent_color"]
        grid_color = STYLE_CONFIG["border_color"]

        self.ax_importance_xgb.set_title("XGBoost Risk Score: Feature Importances", fontdict=title_font_dict)
        if not clear_only and self.xgb_survival_model:
            importances = self.xgb_survival_model.feature_importances_
            feature_names = self.trained_xgb_survival_feature_names
            sorted_indices = np.argsort(importances)[::-1]
            num_plot = min(len(importances), 10)
            self.ax_importance_xgb.barh(range(num_plot), importances[sorted_indices][:num_plot][::-1], align="center", color=bar_color, alpha=0.85)
            self.ax_importance_xgb.set_yticks(range(num_plot))
            self.ax_importance_xgb.set_yticklabels(np.array(feature_names)[sorted_indices][:num_plot][::-1], fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
            self.ax_importance_xgb.set_xlabel("Importance", fontdict=label_font_dict)
            self.ax_importance_xgb.grid(axis='x', linestyle='--', alpha=0.5, color=grid_color)
        
        self.ax_cph_coeffs.set_title("Hybrid CPH Model: Hazard Ratios", fontdict=title_font_dict)
        if not clear_only and self.cph_hybrid_model:
            coeffs = np.exp(self.cph_hybrid_model.params_) 
            conf_int_df = np.exp(self.cph_hybrid_model.confidence_intervals_)
            sorted_coeffs = coeffs.sort_values(ascending=False)
            y_pos = np.arange(len(sorted_coeffs))

            lower_ci_values = conf_int_df.loc[sorted_coeffs.index, '95% lower-bound']
            upper_ci_values = conf_int_df.loc[sorted_coeffs.index, '95% upper-bound']
            errors = [(sorted_coeffs - lower_ci_values).abs().values, (upper_ci_values - sorted_coeffs).abs().values]
            
            self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color, alpha=0.85, xerr=errors, capsize=3, ecolor=tick_color)
            self.ax_cph_coeffs.set_yticks(y_pos)
            self.ax_cph_coeffs.set_yticklabels(sorted_coeffs.index, fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
            self.ax_cph_coeffs.axvline(1.0, color=STYLE_CONFIG["error_text_color"], linestyle='--', linewidth=1.5, alpha=0.8)
            self.ax_cph_coeffs.set_xlabel("Hazard Ratio (HR > 1 is Higher Risk)", fontdict=label_font_dict)
            self.ax_cph_coeffs.grid(axis='x', linestyle='--', alpha=0.5, color=grid_color)

        self.ax_calibration_like.set_title("Hybrid CPH: Predicted Risk Dist (Test)", fontdict=title_font_dict)
        if not clear_only and self.cph_hybrid_model and self.y_test_full_df_for_metrics is not None:
            partial_hazards_test = self.cph_hybrid_model.predict_partial_hazard(self.y_test_full_df_for_metrics)
            event_test = self.y_test_full_df_for_metrics[self.target_event_col_var.get()]
            sns.histplot(partial_hazards_test[event_test == 0], label='Censored (0)', kde=True, ax=self.ax_calibration_like, color="#0097E6", stat="density", bins=20, element="step")
            sns.histplot(partial_hazards_test[event_test == 1], label='Event (1)', kde=True, ax=self.ax_calibration_like, color="#E84118", stat="density", bins=20, element="step")
            self.ax_calibration_like.set_xlabel("Predicted Partial Hazard", fontdict=label_font_dict)
            self.ax_calibration_like.set_ylabel("Density", fontdict=label_font_dict)
            self.ax_calibration_like.legend(fontsize='small')
            self.ax_calibration_like.grid(linestyle='--', alpha=0.5, color=grid_color)
        
        self.ax_survival_curve.set_title("Hybrid CPH: Baseline Survival", fontdict=title_font_dict)
        if not clear_only and self.cph_hybrid_model:
            self.cph_hybrid_model.baseline_survival_.plot(ax=self.ax_survival_curve, color=bar_color, legend=False, linewidth=2)
            self.ax_survival_curve.set_xlabel("Time (Days)", fontdict=label_font_dict)
            self.ax_survival_curve.set_ylabel("Survival Probability", fontdict=label_font_dict)
            self.ax_survival_curve.set_ylim(0, 1.05)
            self.ax_survival_curve.grid(linestyle='--', alpha=0.5, color=grid_color)

        for ax in [self.ax_importance_xgb, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]:
            ax.tick_params(colors=tick_color)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def show_more_plots_window(self):
        if self.more_plots_window is not None and self.more_plots_window.winfo_exists():
            self.more_plots_window.lift()
            return

        if not self.cph_hybrid_model or self.y_test_full_df_for_metrics is None or self.y_test_full_df_for_metrics.empty:
            messagebox.showinfo("No Data", "Hybrid CPH Model must be trained and test data available to view more plots.")
            return

        self.more_plots_window = tk.Toplevel(self.root)
        self.more_plots_window.title("Hybrid CPH Model Diagnostic Plots (Test Set)")
        self.more_plots_window.geometry("950x800")
        self.more_plots_window.configure(bg=STYLE_CONFIG["bg_root"])

        controls_frame = ttk.Frame(self.more_plots_window, style="Content.TFrame")
        controls_frame.pack(pady=5, fill=tk.X)
        ttk.Label(controls_frame, text="Covariate for Partial Effects Plot:").pack(side=tk.LEFT, padx=(10,5))

        available_covariates = list(self.cph_hybrid_model.params_.index)
        self.plot_covariate_var = tk.StringVar()

        self.plot_covariate_selector = ttk.Combobox(
            controls_frame, textvariable=self.plot_covariate_var,
            values=available_covariates, state="readonly", width=30
        )

        if DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME in available_covariates:
            self.plot_covariate_var.set(DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME)
        elif available_covariates:
            self.plot_covariate_var.set(available_covariates[0])
        self.plot_covariate_selector.pack(side=tk.LEFT, padx=5)

        plot_canvas_frame = ttk.Frame(self.more_plots_window)
        plot_canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig_more = plt.Figure(figsize=(9.5, 7.5), dpi=100, facecolor=STYLE_CONFIG["bg_widget"])
        fig_more.subplots_adjust(hspace=0.35, top=0.92, bottom=0.08)
        fig_more.suptitle(
            "Hybrid CPH Diagnostics on Test Set",
            fontsize=STYLE_CONFIG["font_size_section"]+1,
            color=STYLE_CONFIG["fg_header"],
            weight="bold"
        )

        ax_surv_by_risk = fig_more.add_subplot(2, 1, 1)
        ax_partial_effects = fig_more.add_subplot(2, 1, 2)
        canvas_more = FigureCanvasTkAgg(fig_more, master=plot_canvas_frame)

        title_font_dict = {'color': STYLE_CONFIG["fg_header"], 'fontsize': STYLE_CONFIG["font_size_normal"] + 1}
        label_font_dict = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"] - 1}
        tick_color = STYLE_CONFIG["fg_text"]
        grid_color = STYLE_CONFIG["border_color"]

        base_test_df = self.y_test_full_df_for_metrics.copy()
        time_col = self.time_to_event_col_var.get()
        event_col = self.target_event_col_var.get()
        cph_covariates = self.cph_hybrid_model.params_.index.tolist()

        ax_surv_by_risk.clear()
        ax_surv_by_risk.set_title("Survival by Predicted Risk Quartile", fontdict=title_font_dict)

        try:
            from lifelines import KaplanMeierFitter
            df_for_risk_pred = base_test_df[cph_covariates].dropna(subset=cph_covariates)
            partial_hazards = self.cph_hybrid_model.predict_partial_hazard(df_for_risk_pred)

            temp_plot_df = base_test_df.loc[partial_hazards.index].copy()
            temp_plot_df['risk_score'] = partial_hazards.values

            # Split patients into 4 risk groups
            temp_plot_df['risk_group'] = pd.qcut(
                temp_plot_df['risk_score'], 4,
                labels=["Q1 (Lowest Risk)", "Q2 (Moderate-Low)", "Q3 (Moderate-High)", "Q4 (Highest Risk)"],
                duplicates='drop'
            )

            for grp, df_grp in temp_plot_df.groupby('risk_group', observed=True):
                kmf = KaplanMeierFitter()
                kmf.fit(
                    df_grp[time_col],
                    event_observed=df_grp[event_col],
                    label=str(grp)
                )
                kmf.plot_survival_function(ax=ax_surv_by_risk, linewidth=2)

            ax_surv_by_risk.set_xlabel(f"Time ({time_col.split('_')[-1]})", fontdict=label_font_dict)
            ax_surv_by_risk.set_ylabel("Survival Probability", fontdict=label_font_dict)
            ax_surv_by_risk.legend(title="Risk Quartile", facecolor=STYLE_CONFIG["bg_entry"], edgecolor=STYLE_CONFIG["border_color"], labelcolor=tick_color, fontsize='small')
            ax_surv_by_risk.grid(linestyle='--', alpha=0.5, color=grid_color)
        except Exception as e:
            ax_surv_by_risk.text(0.5, 0.5, f"Error plotting survival: {str(e)[:40]}", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])

        ax_surv_by_risk.tick_params(colors=tick_color)
        ax_surv_by_risk.set_facecolor(STYLE_CONFIG["bg_widget"])
        for spine in ax_surv_by_risk.spines.values(): spine.set_color(STYLE_CONFIG["border_color"])

        def redraw_partial_effects_plot(event=None):
            selected = self.plot_covariate_var.get()
            ax_partial_effects.clear()
            ax_partial_effects.set_title(f"Partial Effect of '{selected}'", fontdict=title_font_dict)
            ax_partial_effects.set_facecolor(STYLE_CONFIG["bg_widget"])
            ax_partial_effects.tick_params(colors=tick_color)
            for spine in ax_partial_effects.spines.values(): spine.set_color(STYLE_CONFIG["border_color"])
            ax_partial_effects.grid(linestyle='--', alpha=0.5, color=grid_color)

            if selected in base_test_df.columns:
                cov_series = base_test_df[selected].dropna()
                unique_vals = sorted(cov_series.unique())

                if unique_vals == [0, 1]:
                    values = [0, 1]
                elif len(unique_vals) <= 3:
                    values = unique_vals
                else:
                    pct = np.percentile(cov_series, [10, 50, 90])
                    values = list(np.round(pct, 2))

                try:
                    self.cph_hybrid_model.plot_partial_effects_on_outcome(selected, values=values, ax=ax_partial_effects)
                    ax_partial_effects.set_xlabel(f"Time ({time_col.split('_')[-1]})", fontdict=label_font_dict)
                    ax_partial_effects.set_ylabel("Predicted Survival Probability", fontdict=label_font_dict)
                    if ax_partial_effects.get_legend():
                        ax_partial_effects.legend(fontsize='small', facecolor=STYLE_CONFIG["bg_entry"])
                except Exception as e_pe:
                    ax_partial_effects.text(0.5, 0.5, f"Error: {str(e_pe)[:40]}", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
            else:
                ax_partial_effects.text(0.5, 0.5, "Select a valid covariate", ha='center', va='center', color=tick_color)
            canvas_more.draw_idle()

        self.plot_covariate_selector.bind("<<ComboboxSelected>>", redraw_partial_effects_plot)
        redraw_partial_effects_plot()
        canvas_more.draw()
        canvas_more.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self.more_plots_window, style="Content.TFrame")
        btn_frame.pack(pady=(5,10), fill=tk.X)
        ttk.Button(btn_frame, text="Close", command=self.more_plots_window.destroy).pack()

    def show_about_dialog(self):
        messagebox.showinfo("About", "Advanced Dynamic CVD Predictor V3\n\nDeveloped by ODAT project.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicCVDApp(root)
    root.mainloop()
