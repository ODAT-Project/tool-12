# Developed by ODAT project
# please see https://odat.info
# please see https://github.com/ODAT-Project

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb 
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
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
        self.cph_model = None          
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
        self.prediction_input_widgets = {}
        self.dynamic_input_scrollable_frame = None
        self.more_plots_window = None
        self.current_report_lines = []

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
            print(message)
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
        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
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

        self.assess_button = ttk.Button(self.dynamic_input_scrollable_frame, text="Assess Risk (CPH)", command=self.assess_risk_action)
        self.assess_button.pack(pady=(15,10), fill=tk.X, padx=5)
        self.dynamic_input_scrollable_frame.update_idletasks()
        self.dynamic_input_canvas.config(scrollregion=self.dynamic_input_canvas.bbox("all"))

    def create_results_display_widgets(self, parent_frame):
        top_frame = ttk.Frame(parent_frame, style="Content.TFrame")
        top_frame.pack(fill=tk.X, pady=5)

        pred_res_frame = ttk.LabelFrame(top_frame, text="CPH Prediction Result")
        pred_res_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        self.risk_prob_label = ttk.Label(pred_res_frame, text="N/A", font=(STYLE_CONFIG["font_family"], 22, "bold"), foreground=STYLE_CONFIG["accent_color"])
        self.risk_prob_label.pack(pady=(5,2))
        self.risk_interpretation_label = ttk.Label(pred_res_frame, text="Train model & assess.")
        self.risk_interpretation_label.pack(pady=(0,5))

        btn_container = ttk.Frame(top_frame, style="Content.TFrame")
        btn_container.pack(side=tk.RIGHT, padx=(5,0), pady=10, anchor="ne")

        self.save_results_button = ttk.Button(btn_container, text="Save Report & Plots", command=self.save_results_action, state=tk.DISABLED)
        self.save_results_button.pack(side=tk.TOP, fill=tk.X, pady=(0,5))

        self.more_plots_button = ttk.Button(btn_container, text="View Survival Plots", command=self.show_more_plots_window, state=tk.DISABLED)
        self.more_plots_button.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.LabelFrame(parent_frame, text="Model Performance Visuals")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Matplotlib Styling
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

        if hasattr(self, 'train_button'): self.train_button.config(state=train_state)
        if hasattr(self, 'xgb_feature_listbox'): self.xgb_feature_listbox.config(state=train_state)
        if hasattr(self, 'cph_linear_feature_listbox'): self.cph_linear_feature_listbox.config(state=train_state)

        if hasattr(self, 'target_event_selector'): self.target_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        if hasattr(self, 'time_to_event_selector'): self.time_to_event_selector.config(state="readonly" if data_loaded else tk.DISABLED)
        if hasattr(self, 'time_horizon_entry'): self.time_horizon_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)

        if hasattr(self, 'n_estimators_entry'): self.n_estimators_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        if hasattr(self, 'max_depth_entry'): self.max_depth_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        if hasattr(self, 'learning_rate_entry'): self.learning_rate_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        if hasattr(self, 'cph_penalizer_entry'): self.cph_penalizer_entry.config(state=tk.NORMAL if data_loaded else tk.DISABLED)

        if hasattr(self, 'assess_button') and self.assess_button.winfo_exists():
            self.assess_button.config(state=predict_state)
        if hasattr(self, 'more_plots_button'): self.more_plots_button.config(state=predict_state)
        if hasattr(self, 'save_results_button'): self.save_results_button.config(state=predict_state)

        for _feature_name, widget in self.prediction_input_widgets.items():
            if hasattr(widget, 'config'):
                widget.config(state=tk.NORMAL if model_trained else tk.DISABLED)

    def downcast_numerics(self, df):
        self.log_training_message("  Attempting to downcast numeric types for memory optimization...")
        f_cols = df.select_dtypes('float').columns
        i_cols = df.select_dtypes('integer').columns
        df[f_cols] = df[f_cols].apply(pd.to_numeric, downcast='float')
        df[i_cols] = df[i_cols].apply(pd.to_numeric, downcast='integer')
        gc.collect()
        return df

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
        if not filepath:
            self.log_training_message("File loading cancelled by user.")
            return
        try:
            self.data_df = pd.read_csv(filepath, low_memory=False)
            self.data_df = self.downcast_numerics(self.data_df.copy())
            gc.collect() 
            self.loaded_file_label.config(text=f"Loaded: {filepath.split('/')[-1]} ({self.data_df.shape[0]} rows, {self.data_df.shape[1]} cols)")
            self.log_training_message(f"Successfully loaded and downcasted {filepath.split('/')[-1]}.")
            
            column_names = sorted([str(col) for col in self.data_df.columns if str(col).strip()])
            if not column_names:
                self.log_training_message("No columns found in CSV or header is missing.", is_error=True)
                messagebox.showerror("CSV Error", "No columns detected in CSV.")
                self.toggle_train_predict_sections_enabled(data_loaded=False, model_trained=False)
                return

            self.root.after(10, self._populate_ui_lists_after_load, column_names)

            self.xgb_survival_model = None
            self.cph_model = None
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=False)
            
            if self.dynamic_input_scrollable_frame:
                for widget in self.dynamic_input_scrollable_frame.winfo_children(): widget.destroy()
            self.placeholder_pred_label = ttk.Label(self.dynamic_input_scrollable_frame, text="Train a model to enable prediction inputs.")
            self.placeholder_pred_label.pack(padx=10, pady=20)
            if hasattr(self, 'assess_button') and self.assess_button.winfo_exists(): self.assess_button.pack_forget()

            self.update_plots(clear_only=True)
            self.risk_interpretation_label.config(text="Data loaded. Configure and train model.")
            self.risk_prob_label.config(text="N/A")

        except Exception as e:
            self.log_training_message(f"Error loading CSV: {str(e)}", is_error=True)
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

    def train_model_action(self):
        if self.data_df is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        selected_xgb_indices = self.xgb_feature_listbox.curselection()
        selected_cph_linear_indices = self.cph_linear_feature_listbox.curselection()
        target_event_col = self.target_event_col_var.get()
        time_to_event_col = self.time_to_event_col_var.get()

        if not selected_xgb_indices:
            messagebox.showerror("Error", "No features selected for XGBoost Risk Score.")
            return

        if not target_event_col or not time_to_event_col:
            messagebox.showerror("Error", "Target event and time-to-event columns must be selected.")
            return

        self.trained_xgb_survival_feature_names = [self.xgb_feature_listbox.get(i) for i in selected_xgb_indices]
        self.trained_cph_linear_feature_names = [self.cph_linear_feature_listbox.get(i) for i in selected_cph_linear_indices]

        for col_list_name, col_list in [("XGB", self.trained_xgb_survival_feature_names), ("CPH", self.trained_cph_linear_feature_names)]:
            if target_event_col in col_list or time_to_event_col in col_list:
                messagebox.showerror("Error", f"Target/Time column cannot be in {col_list_name} feature list.")
                return
        
        if DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME in self.trained_cph_linear_feature_names:
             messagebox.showerror("Error", f"'{DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME}' is a reserved name for the XGBoost score.")
             return

        try:
            n_estimators = int(self.n_estimators_var.get())
            max_depth = int(self.max_depth_var.get())
            learning_rate = float(self.learning_rate_var.get())
            cph_penalizer = float(self.cph_penalizer_var.get())
            if n_estimators <=0 or max_depth <=0 or not (0 < learning_rate <= 1) or cph_penalizer < 0:
                raise ValueError("Hyperparameters out of valid range.")
        except ValueError:
            messagebox.showerror("Hyperparameter Error", "Invalid hyperparameter values.")
            return

        self.log_training_message("--- Starting Hybrid Model Training ---")
        full_df_processed = self.data_df.copy()
        gc.collect() 

        try:
            full_df_processed[target_event_col] = pd.to_numeric(full_df_processed[target_event_col], errors='raise')
            full_df_processed[time_to_event_col] = pd.to_numeric(full_df_processed[time_to_event_col], errors='raise')
        except Exception as e:
            self.log_training_message(f"Error: Target or Time column contains non-numeric values: {e}", is_error=True)
            messagebox.showerror("Data Error", f"Target or Time column has non-numeric data: {e}")
            return

        #cleaning and splitting before training
        initial_rows = len(full_df_processed)
        full_df_processed.dropna(subset=[target_event_col, time_to_event_col], inplace=True)
        full_df_processed = full_df_processed[full_df_processed[time_to_event_col] > 0]
        
        if len(full_df_processed) < initial_rows:
            self.log_training_message(f"  Dropped {initial_rows - len(full_df_processed)} rows due to NaNs or non-positive times.")
        
        if full_df_processed.empty:
            messagebox.showerror("Data Error", "No valid data remaining.")
            return

        train_df, test_df = train_test_split(
            full_df_processed, 
            test_size=0.25, 
            random_state=42,
            stratify=full_df_processed[target_event_col] if full_df_processed[target_event_col].nunique() > 1 else None
        )

        #xgboost on training data only
        self.log_training_message("\n--- Stage 1: Training XGBoost Survival Model for Risk Score ---")
        try:
            self.scaler_xgb = StandardScaler()
            self.num_imputer_xgb = SimpleImputer(strategy='median')
            self.trained_feature_medians_xgb = {}
            self.scaled_columns_xgb = []

            X_xgb_train = self._preprocess_features(
                train_df, self.trained_xgb_survival_feature_names,
                self.trained_feature_medians_xgb, self.scaler_xgb,
                self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=True
            )

            X_xgb_test = self._preprocess_features(
                test_df, self.trained_xgb_survival_feature_names,
                self.trained_feature_medians_xgb, self.scaler_xgb,
                self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=False
            )

            y_train_event = train_df[target_event_col].astype(int)
            y_train_time = train_df[time_to_event_col]
            y_xgb_survival_train = np.where(y_train_event == 1, y_train_time, -y_train_time)

            self.xgb_survival_model = xgb.XGBModel(
                objective='survival:cox',
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            
            self.xgb_survival_model.fit(X_xgb_train, y_xgb_survival_train)
            self.log_training_message("  XGBoost survival model trained.")

            train_xgb_scores = self.xgb_survival_model.predict(X_xgb_train, output_margin=True)
            test_xgb_scores = self.xgb_survival_model.predict(X_xgb_test, output_margin=True)

            df_cph_train_fitting = pd.DataFrame(index=train_df.index)
            df_cph_test_fitting = pd.DataFrame(index=test_df.index)
            
            df_cph_train_fitting[DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME] = train_xgb_scores
            df_cph_test_fitting[DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME] = test_xgb_scores

        except Exception as e:
            self.log_training_message(f"Error in XGBoost Survival training: {str(e)}", is_error=True)
            messagebox.showerror("XGBoost Training Error", f"Failed during XGBoost stage: {e}")
            self.xgb_survival_model = None
            return

        #CPH on training data only
        self.log_training_message("\n--- Stage 2: Training Cox Proportional Hazards (CPH) Model ---")
        try:
            if self.trained_cph_linear_feature_names:
                self.scaler_cph_linear = StandardScaler()
                self.num_imputer_cph_linear = SimpleImputer(strategy='median')
                self.trained_feature_medians_cph_linear = {}
                self.scaled_columns_cph_linear = []

                X_cph_linear_train = self._preprocess_features(
                    train_df, self.trained_cph_linear_feature_names,
                    self.trained_feature_medians_cph_linear, self.scaler_cph_linear,
                    self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=True
                )
                
                X_cph_linear_test = self._preprocess_features(
                    test_df, self.trained_cph_linear_feature_names,
                    self.trained_feature_medians_cph_linear, self.scaler_cph_linear,
                    self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False
                )

                for col in X_cph_linear_train.columns:
                    df_cph_train_fitting[col] = X_cph_linear_train[col]
                    df_cph_test_fitting[col] = X_cph_linear_test[col]

            df_cph_train_fitting[time_to_event_col] = train_df[time_to_event_col].values
            df_cph_train_fitting[target_event_col] = train_df[target_event_col].astype(int).values
            
            df_cph_test_fitting[time_to_event_col] = test_df[time_to_event_col].values
            df_cph_test_fitting[target_event_col] = test_df[target_event_col].astype(int).values

            self.cph_model = CoxPHFitter(penalizer=cph_penalizer, l1_ratio=0.0) 
            self.cph_model.fit(df_cph_train_fitting, duration_col=time_to_event_col, event_col=target_event_col)
            
            self.log_training_message("  CPH model trained.")
            self.log_training_message(f"  CPH Concordance on training: {self.cph_model.concordance_index_:.4f}")

            self.log_training_message("\n  Checking Proportional Hazards Assumption (on training data):")
            try:
                results_df = self.cph_model.check_assumptions(df_cph_train_fitting, p_value_threshold=0.05, show_plots=False)
                self.log_training_message("    PH Assumption Test p-values:")
                for covariate, p_value in results_df['p'].items():
                    self.log_training_message(f"      {covariate}: {p_value:.4f} {'(Potential Violation)' if p_value < 0.05 else ''}")
            except Exception as e_ph:
                self.log_training_message(f"    Error during PH assumption check: {e_ph}", is_error=True)

            self.y_test_full_df_for_metrics = df_cph_test_fitting.copy()
            
            if not df_cph_test_fitting.empty:
                cph_test_c_index = self.cph_model.score(df_cph_test_fitting, scoring_method="concordance_index")
                self.log_training_message(f"  CPH Concordance on test set: {cph_test_c_index:.4f}")
            else:
                 self.log_training_message("  Test set for CPH is empty, cannot calculate test C-index.", is_error=True)

            self.generate_training_report(target_event_col, time_to_event_col, n_estimators, max_depth, learning_rate, cph_penalizer)
            self.create_dynamic_prediction_inputs() 
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=True)
            self.update_plots()

        except Exception as e:
            self.log_training_message(f"Error in CPH training: {str(e)}", is_error=True)
            messagebox.showerror("CPH Training Error", f"Failed during CPH stage: {e}")
            self.cph_model = None
            self.toggle_train_predict_sections_enabled(data_loaded=True, model_trained=False) 
        finally:
            del full_df_processed
            if 'df_cph_train_fitting' in locals(): del df_cph_train_fitting
            if 'X_xgb_train' in locals(): del X_xgb_train
            gc.collect() 

    def generate_training_report(self, target_col, time_col, n_est, max_d, lr, cph_pen):
        self.current_report_lines = [] 
        self.current_report_lines.append("\n--- Combined Model Training Report ---")
        self.current_report_lines.append(f"Dataset Shape (after initial cleaning): {self.data_df.shape if self.data_df is not None else 'N/A'}")
        self.current_report_lines.append(f"Target Event Column: {target_col}, Time Column: {time_col}")
        self.current_report_lines.append("\nXGBoost Survival Model (for Risk Score):")
        self.current_report_lines.append(f"  Features used: {len(self.trained_xgb_survival_feature_names)}")
        self.current_report_lines.append(f"  Hyperparameters: Estimators={n_est}, MaxDepth={max_d}, LearningRate={lr}")
        self.current_report_lines.append("\nCPH Model:")
        cph_features_in_model = self.trained_cph_linear_feature_names + [DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME]
        self.current_report_lines.append(f"  Features used (incl. XGBoost score): {len(cph_features_in_model)}")
        self.current_report_lines.append(f"  Linear CPH Features: {self.trained_cph_linear_feature_names}")
        self.current_report_lines.append(f"  CPH L2 Penalizer: {cph_pen}")

        if self.cph_model and hasattr(self.cph_model, 'summary') and not self.cph_model.summary.empty:
            self.current_report_lines.append("\nCPH Model Summary (Hazard Ratios):")
            summary_df_to_print = self.cph_model.summary.reset_index()
            original_index_name = self.cph_model.summary.index.name
            
            if original_index_name is None and 'index' in summary_df_to_print.columns:
                original_index_name = 'index'
                summary_df_to_print.rename(columns={'index': 'covariate'}, inplace=True) 
                original_index_name = 'covariate' 
            elif original_index_name not in summary_df_to_print.columns and 'index' in summary_df_to_print.columns :
                 summary_df_to_print.rename(columns={'index': original_index_name or 'covariate'}, inplace=True)
                 if not original_index_name: original_index_name = 'covariate'

            summary_cols_from_user_code = ['exp(coef)', 'exp(coef)_lower 95%', 'exp(coef)_upper 95%', 'p']
            actual_lifelines_summary_cols = {
                'exp(coef)': 'exp(coef)',
                'exp(coef)_lower 95%': 'exp(coef) lower 95%', 
                'exp(coef)_upper 95%': 'exp(coef) upper 95%',
                'p': 'p'
            }
            
            columns_for_summary_text = [original_index_name] 
            for col_key in summary_cols_from_user_code:
                if actual_lifelines_summary_cols.get(col_key) in summary_df_to_print.columns:
                    columns_for_summary_text.append(actual_lifelines_summary_cols.get(col_key))
                elif col_key in summary_df_to_print.columns:
                     columns_for_summary_text.append(col_key)

            final_cols_to_print_for_summary = [col for col in columns_for_summary_text if col in summary_df_to_print.columns]

            if final_cols_to_print_for_summary:
                summary_text = summary_df_to_print[final_cols_to_print_for_summary].to_string(index=False)
                self.current_report_lines.extend(summary_text.split('\n'))
            else:
                self.current_report_lines.append("  Could not format CPH model summary (column names mismatch).")
        else:
            self.current_report_lines.append("  CPH model summary not available.")

        if self.y_test_full_df_for_metrics is not None and not self.y_test_full_df_for_metrics.empty and self.cph_model:
            try:
                cph_test_c_index = self.cph_model.score(self.y_test_full_df_for_metrics, scoring_method="concordance_index")
                self.current_report_lines.append(f"\nCPH Performance on Test Set:")
                self.current_report_lines.append(f"  Concordance Index (C-index): {cph_test_c_index:.4f}")
            except Exception as e:
                self.current_report_lines.append(f"  Could not calculate CPH test C-index: {e}") 

        self.current_report_lines.append("--- End of Report ---")
        self.current_report_lines.append("-----------------------")
        self.current_report_lines.append("--- by ODAT project ---")

        # Output to GUI
        for line in self.current_report_lines:
            self.log_training_message(line, is_error=False)

    def save_results_action(self):
        if not self.current_report_lines:
            messagebox.showinfo("Nothing to Save", "Train the model first to generate a report.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            title="Save Training Report & Plots",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save text report
                with open(file_path, "w", encoding='utf-8') as f:
                    for line in self.current_report_lines:
                        f.write(line + "\n")
                
                # Save plot figure
                fig_path = file_path.rsplit('.', 1)[0] + "_plots.png"
                self.fig.savefig(fig_path, dpi=300, bbox_inches='tight')

                messagebox.showinfo("Success", f"Saved successfully:\n\nReport: {file_path}\nPlots: {fig_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")

    def assess_risk_action(self):
        if not self.xgb_survival_model or not self.cph_model or not self.all_base_features_for_input:
            messagebox.showerror("Error", "Model not fully trained or features unclear.")
            return

        input_values_from_gui = {}
        try:
            for feature_name, widget in self.prediction_input_widgets.items():
                value_str = widget.get()
                try: input_values_from_gui[feature_name] = float(value_str)
                except ValueError:
                    if value_str.lower() in ["true", "yes", "1", "male"]: input_values_from_gui[feature_name] = 1.0
                    elif value_str.lower() in ["false", "no", "0", "female"]: input_values_from_gui[feature_name] = 0.0
                    else:
                        self.log_training_message(f"Warning: Non-numeric/non-binary input for '{feature_name}': '{value_str}'. Will be imputed as NaN.", is_error=True)
                        input_values_from_gui[feature_name] = np.nan
        except Exception as e:
            messagebox.showerror("Input Error", f"Error reading input values: {e}")
            return

        input_df_xgb = pd.DataFrame([input_values_from_gui])[self.trained_xgb_survival_feature_names]
        input_xgb_processed = self._preprocess_features(
            input_df_xgb, self.trained_xgb_survival_feature_names,
            self.trained_feature_medians_xgb, self.scaler_xgb,
            self.scaled_columns_xgb, self.num_imputer_xgb, fit_transform=False 
        )

        try:
            new_xgb_risk_score = self.xgb_survival_model.predict(input_xgb_processed, output_margin=True)[0]
        except Exception as e:
            self.log_training_message(f"XGBoost prediction error for risk score: {str(e)}", is_error=True)
            messagebox.showerror("Prediction Error", f"Could not generate XGBoost risk score: {e}")
            return

        cph_input_dict = {DynamicCVDApp.XGBOOST_RISK_SCORE_COL_NAME: new_xgb_risk_score}
        
        if self.trained_cph_linear_feature_names:
            input_df_cph_linear = pd.DataFrame([input_values_from_gui])[self.trained_cph_linear_feature_names]
            input_cph_linear_processed = self._preprocess_features(
                input_df_cph_linear, self.trained_cph_linear_feature_names,
                self.trained_feature_medians_cph_linear, self.scaler_cph_linear,
                self.scaled_columns_cph_linear, self.num_imputer_cph_linear, fit_transform=False
            )
            for col in input_cph_linear_processed.columns:
                cph_input_dict[col] = input_cph_linear_processed[col].iloc[0]

        input_data_for_cph_model = pd.DataFrame([cph_input_dict])
        cph_model_cols = self.cph_model.summary.index.tolist()
        input_data_for_cph_model = input_data_for_cph_model[cph_model_cols]

        try:
            time_horizon_str = self.time_horizon_var.get().strip()
            pred_horizon_years = 5 
            
            if time_horizon_str:
                try:
                    pred_horizon_years = float(time_horizon_str)
                    if pred_horizon_years <= 0: pred_horizon_years = 5 
                except ValueError:
                    self.log_training_message(f"Invalid prediction horizon '{time_horizon_str}', using 5 years.", is_error=True)

            pred_horizon_days = pred_horizon_years * 365.25  
            survival_prob_at_horizon = self.cph_model.predict_survival_function(input_data_for_cph_model, times=[pred_horizon_days]).iloc[0,0]
            
            risk_at_horizon = 1.0 - survival_prob_at_horizon

            self.risk_prob_label.config(text=f"{pred_horizon_years:.0f}-Year Risk: {risk_at_horizon*100:.1f}%")

            if risk_at_horizon > 0.20: 
                self.risk_interpretation_label.config(text="Higher Risk", foreground=STYLE_CONFIG["error_text_color"])
            elif risk_at_horizon > 0.10:
                self.risk_interpretation_label.config(text="Moderate Risk", foreground="#F39C12") 
            else:
                self.risk_interpretation_label.config(text="Lower Risk", foreground="#27AE60") 

        except Exception as e:
            self.log_training_message(f"CPH prediction error: {str(e)}", is_error=True)
            messagebox.showerror("Prediction Error", f"Could not make CPH prediction: {e}")
            self.risk_prob_label.config(text="Error"); self.risk_interpretation_label.config(text="")
        finally:
            del input_df_xgb, input_xgb_processed, input_data_for_cph_model
            if 'input_df_cph_linear' in locals(): del input_df_cph_linear
            if 'input_cph_linear_processed' in locals(): del input_cph_linear_processed
            gc.collect()

    def update_plots(self, clear_only=False):
        ax_list = [self.ax_importance_xgb, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]
        for ax in ax_list: ax.clear()

        title_font_dict = {'color': STYLE_CONFIG["fg_header"], 'fontsize': STYLE_CONFIG["font_size_normal"] + 1, 'weight': 'bold'}
        label_font_dict = {'color': STYLE_CONFIG["fg_text"], 'fontsize': STYLE_CONFIG["font_size_normal"] - 1}
        tick_color = STYLE_CONFIG["fg_text"]
        bar_color = STYLE_CONFIG["accent_color"]
        grid_color = STYLE_CONFIG["border_color"]

        self.ax_importance_xgb.set_title("XGBoost Risk Score: Feature Importances", fontdict=title_font_dict)
        if clear_only or not self.xgb_survival_model or not hasattr(self.xgb_survival_model, 'feature_importances_') or not self.trained_xgb_survival_feature_names:
            self.ax_importance_xgb.text(0.5, 0.5, "Train model to view importances.", ha='center', va='center', color=tick_color)
        else:
            importances = self.xgb_survival_model.feature_importances_
            feature_names_for_plot = self.trained_xgb_survival_feature_names
            if len(importances) == len(feature_names_for_plot) and len(importances) > 0:
                sorted_indices = np.argsort(importances)[::-1]
                num_features_to_plot = min(len(importances), 10)
                self.ax_importance_xgb.barh(range(num_features_to_plot), importances[sorted_indices][:num_features_to_plot][::-1], align="center", color=bar_color, alpha=0.85)
                self.ax_importance_xgb.set_yticks(range(num_features_to_plot))
                self.ax_importance_xgb.set_yticklabels(np.array(feature_names_for_plot)[sorted_indices][:num_features_to_plot][::-1], fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
                self.ax_importance_xgb.set_xlabel("Importance", fontdict=label_font_dict)
                self.ax_importance_xgb.grid(axis='x', linestyle='--', alpha=0.5, color=grid_color)
            else:
                 self.ax_importance_xgb.text(0.5, 0.5, "Importances data error.", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        
        self.ax_importance_xgb.tick_params(axis='x', colors=tick_color); self.ax_importance_xgb.tick_params(axis='y', colors=tick_color)

        self.ax_cph_coeffs.set_title("CPH Model: Log(Hazard Ratios)", fontdict=title_font_dict)
        if clear_only or not self.cph_model or self.cph_model.summary.empty:
            self.ax_cph_coeffs.text(0.5, 0.5, "Train model to view CPH coefficients.", ha='center', va='center', color=tick_color)
        else:
            coeffs = self.cph_model.params_ 
            conf_int_df = self.cph_model.confidence_intervals_

            if not coeffs.empty and not conf_int_df.empty:
                sorted_coeffs = coeffs.sort_values(ascending=False)
                y_pos = np.arange(len(sorted_coeffs))

                lower_col_name = '95% lower-bound'
                upper_col_name = '95% upper-bound'

                if lower_col_name in conf_int_df.columns and upper_col_name in conf_int_df.columns:
                    valid_indices_for_ci = sorted_coeffs.index.intersection(conf_int_df.index)
                    if len(valid_indices_for_ci) == len(sorted_coeffs):
                        lower_ci_values = conf_int_df.loc[sorted_coeffs.index, lower_col_name]
                        upper_ci_values = conf_int_df.loc[sorted_coeffs.index, upper_col_name]
                        errors = [
                            (sorted_coeffs - lower_ci_values).abs().values,
                            (upper_ci_values - sorted_coeffs).abs().values
                        ]
                        self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color, alpha=0.85, xerr=errors, capsize=3, ecolor=tick_color)
                    else:
                        self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color, alpha=0.85)
                else:
                    self.ax_cph_coeffs.barh(y_pos, sorted_coeffs.values, align='center', color=bar_color, alpha=0.85)

                self.ax_cph_coeffs.set_yticks(y_pos)
                self.ax_cph_coeffs.set_yticklabels(sorted_coeffs.index, fontdict={'fontsize': STYLE_CONFIG["font_size_normal"]-2})
                self.ax_cph_coeffs.axvline(0, color=STYLE_CONFIG["error_text_color"], linestyle='--', linewidth=1, alpha=0.7)
                self.ax_cph_coeffs.set_xlabel("Log(Hazard Ratio)", fontdict=label_font_dict)
                self.ax_cph_coeffs.grid(axis='x', linestyle='--', alpha=0.5, color=grid_color)
            else:
                self.ax_cph_coeffs.text(0.5, 0.5, "CPH coefficients not available.", ha='center', va='center', color=tick_color)

        self.ax_cph_coeffs.tick_params(axis='x', colors=tick_color); self.ax_cph_coeffs.tick_params(axis='y', colors=tick_color)

        self.ax_calibration_like.set_title("CPH: Predicted Risk Distribution (Test)", fontdict=title_font_dict)
        if clear_only or not self.cph_model or self.y_test_full_df_for_metrics is None or self.y_test_full_df_for_metrics.empty:
            self.ax_calibration_like.text(0.5, 0.5, "Train model to view risk distribution.", ha='center', va='center', color=tick_color)
        else:
            try:
                partial_hazards_test = self.cph_model.predict_partial_hazard(self.y_test_full_df_for_metrics)
                event_test = self.y_test_full_df_for_metrics[self.target_event_col_var.get()]

                sns.histplot(partial_hazards_test[event_test == 0], label='Censored (0)', kde=True, ax=self.ax_calibration_like, color="#0097E6", stat="density", bins=20, element="step")
                sns.histplot(partial_hazards_test[event_test == 1], label='Event (1)', kde=True, ax=self.ax_calibration_like, color="#E84118", stat="density", bins=20, element="step")
                
                self.ax_calibration_like.set_xlabel("Predicted Partial Hazard (Risk Score)", fontdict=label_font_dict)
                self.ax_calibration_like.set_ylabel("Density", fontdict=label_font_dict)
                self.ax_calibration_like.legend(facecolor=STYLE_CONFIG["bg_entry"], edgecolor=STYLE_CONFIG["border_color"], labelcolor=tick_color, framealpha=1.0)
                self.ax_calibration_like.grid(linestyle='--', alpha=0.5, color=grid_color)
            except Exception as e:
                self.ax_calibration_like.text(0.5, 0.5, f"Error: {str(e)[:30]}...", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])
        
        self.ax_calibration_like.tick_params(colors=tick_color)

        self.ax_survival_curve.set_title("CPH: Baseline Survival Function", fontdict=title_font_dict)
        if clear_only or not self.cph_model or not hasattr(self.cph_model, 'baseline_survival_') or self.cph_model.baseline_survival_.empty:
            self.ax_survival_curve.text(0.5, 0.5, "Train CPH model to view baseline survival.", ha='center', va='center', color=tick_color)
        else:
            try:
                self.cph_model.baseline_survival_.plot(ax=self.ax_survival_curve, color=bar_color, legend=False, linewidth=2)
                self.ax_survival_curve.set_xlabel(f"Time ({self.time_to_event_col_var.get().split('_')[-1] if self.time_to_event_col_var.get() else 'Days'})", fontdict=label_font_dict)
                self.ax_survival_curve.set_ylabel("Survival Probability", fontdict=label_font_dict)
                self.ax_survival_curve.set_ylim(0, 1.05)
                self.ax_survival_curve.grid(linestyle='--', alpha=0.5, color=grid_color)
            except Exception as e:
                 self.ax_survival_curve.text(0.5, 0.5, f"Error plotting: {str(e)[:30]}", ha='center', va='center', color=STYLE_CONFIG["error_text_color"])

        self.ax_survival_curve.tick_params(colors=tick_color)

        for ax in [self.ax_importance_xgb, self.ax_cph_coeffs, self.ax_calibration_like, self.ax_survival_curve]:
            for spine in ax.spines.values():
                spine.set_color(STYLE_CONFIG["border_color"])

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()
        gc.collect() 

    def show_more_plots_window(self):
        if self.more_plots_window is not None and self.more_plots_window.winfo_exists():
            self.more_plots_window.lift()
            return
        if not self.cph_model or self.y_test_full_df_for_metrics is None or self.y_test_full_df_for_metrics.empty:
            messagebox.showinfo("No Data", "CPH Model must be trained and test data available to view more plots.")
            return

        self.more_plots_window = tk.Toplevel(self.root)
        self.more_plots_window.title("CPH Model Diagnostic Plots (Test Set)")
        self.more_plots_window.geometry("950x800")
        self.more_plots_window.configure(bg=STYLE_CONFIG["bg_root"])

        controls_frame = ttk.Frame(self.more_plots_window, style="Content.TFrame")
        controls_frame.pack(pady=5, fill=tk.X)
        ttk.Label(controls_frame, text="Covariate for Partial Effects Plot:").pack(side=tk.LEFT, padx=(10,5))

        available_covariates = list(self.cph_model.params_.index)
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
            "CPH Model Diagnostics on Test Set",
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
        cph_covariates = self.cph_model.params_.index.tolist()

        ax_surv_by_risk.clear()
        ax_surv_by_risk.set_title("Survival by Predicted Risk Quartile", fontdict=title_font_dict)
        
        try:
            from lifelines import KaplanMeierFitter
            df_for_risk_pred = base_test_df[cph_covariates].dropna(subset=cph_covariates)
            partial_hazards = self.cph_model.predict_partial_hazard(df_for_risk_pred)
            
            temp_plot_df = base_test_df.loc[partial_hazards.index].copy()
            temp_plot_df['risk_score'] = partial_hazards.values
            temp_plot_df['risk_group'] = pd.qcut(
                temp_plot_df['risk_score'], 4,
                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates='drop'
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
            ax_surv_by_risk.legend(
                title="Risk Quartile",
                facecolor=STYLE_CONFIG["bg_entry"],
                edgecolor=STYLE_CONFIG["border_color"],
                labelcolor=tick_color,
                fontsize='small'
            )
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
                    self.cph_model.plot_partial_effects_on_outcome(selected, values=values, ax=ax_partial_effects)
                    ax_partial_effects.set_xlabel(f"Time ({time_col.split('_')[-1]})", fontdict=label_font_dict)
                    ax_partial_effects.set_ylabel("Predicted Survival Probability", fontdict=label_font_dict)
                    if ax_partial_effects.get_legend():
                        ax_partial_effects.legend(fontsize='small')
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
        messagebox.showinfo("About Advanced Dynamic CVD Predictor",
                            "Advanced Dynamic CVD Predictor -- XGBoost-CPH V2\n\n"
                            "XGBoost Survival model generates a non-linear risk score. CPH model incorporates this score along with other linear predictors for survival analysis.\n\n"
                            "Developed by ODAT project.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DynamicCVDApp(root)
    root.mainloop()
