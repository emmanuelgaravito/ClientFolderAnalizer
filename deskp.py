import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Set
import threading

class ClientFolderAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.initialize_gui()
        self.connection_status = False
        self.selected_clients: Set[str] = set()
        self.processing = False
        self.output_df = pd.DataFrame()
        
    def setup_logging(self):
        """Configure logging to both file and console"""
        log_filename = f'client_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def initialize_gui(self):
        """Set up the GUI with client selection capabilities"""
        self.window = tk.Tk()
        self.window.title("Client Folder Analyzer")
        self.window.geometry("1000x800")
        
        # Apply styles
        style = ttk.Style()
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        
        # Create main container with two frames
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for folder selection and controls
        left_frame = ttk.Frame(self.main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Right frame for client list
        right_frame = ttk.Frame(self.main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Status and controls in left frame
        self.status_label = ttk.Label(left_frame, text="Status: Not Connected", style='Error.TLabel')
        self.status_label.pack(pady=5)
        
        self.folder_label = ttk.Label(left_frame, text="No root folder selected")
        self.folder_label.pack(pady=5)
        
        ttk.Button(left_frame, text="Select Root Folder", command=self.select_root_folder).pack(pady=5)
        ttk.Button(left_frame, text="Refresh Client List", command=self.refresh_client_list).pack(pady=5)
        ttk.Button(left_frame, text="Process Selected Clients", command=self.process_selected_clients).pack(pady=5)
        
        # Progress section
        self.progress_frame = ttk.Frame(left_frame)
        self.progress_frame.pack(pady=10, fill=tk.X)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack()
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, 
            length=300, 
            mode='determinate', 
            variable=self.progress_var
        )
        self.progress_bar.pack(pady=5)
        
        # Results text area
        self.results_text = tk.Text(left_frame, height=20, width=50)
        self.results_text.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Client list in right frame
        right_frame_label = ttk.Label(right_frame, text="Available Clients")
        right_frame_label.pack(pady=5)
        
        # Search box for clients
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_client_list)
        search_entry = ttk.Entry(right_frame, textvariable=self.search_var)
        search_entry.pack(pady=5, fill=tk.X)
        
        # Client listbox with scrollbar
        self.listbox_frame = ttk.Frame(right_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.client_listbox = tk.Listbox(
            self.listbox_frame, 
            selectmode=tk.MULTIPLE, 
            height=20
        )
        self.client_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.listbox_frame, orient="vertical", command=self.client_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.client_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Batch selection buttons
        batch_frame = ttk.Frame(right_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(batch_frame, text="Select First 10", command=lambda: self.select_batch(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Select First 50", command=lambda: self.select_batch(50)).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=2)

    def select_root_folder(self):
        """Select root folder containing client folders"""
        folder = filedialog.askdirectory()
        if folder:
            self.root_path = Path(folder)
            self.folder_label.configure(text=str(self.root_path))
            self.verify_cloud_connection()
            self.refresh_client_list()

    def verify_cloud_connection(self):
        """Verify if the selected folder is in cloud storage"""
        try:
            if not hasattr(self, 'root_path'):
                raise ValueError("No folder selected")
                
            if not self.root_path.exists():
                raise ValueError("Selected folder does not exist")
                
            # Check if path contains common cloud storage indicators
            cloud_indicators = ['Google Drive', 'OneDrive', 'Dropbox']
            path_str = str(self.root_path).lower()
            
            self.connection_status = any(indicator.lower() in path_str for indicator in cloud_indicators)
            
            if self.connection_status:
                self.status_label.configure(text="Status: Connected to cloud storage", style='Success.TLabel')
                logging.info(f"Connected to cloud storage at: {self.root_path}")
            else:
                self.status_label.configure(text="Status: Local folder (not cloud storage)", style='Error.TLabel')
                logging.warning("Selected folder is not in cloud storage")
                
        except Exception as e:
            self.connection_status = False
            self.status_label.configure(text=f"Status: Error - {str(e)}", style='Error.TLabel')
            logging.error(f"Connection verification failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to verify folder access: {str(e)}")

    def refresh_client_list(self):
        """Refresh the list of available client folders"""
        if not hasattr(self, 'root_path'):
            messagebox.showwarning("Warning", "Please select a root folder first")
            return
            
        try:
            # Get all immediate subdirectories (client folders)
            self.client_folders = [
                f.name for f in self.root_path.iterdir() 
                if f.is_dir() and not f.name.startswith('.')
            ]
            
            self.client_folders.sort()
            self.update_client_listbox(self.client_folders)
            logging.info(f"Found {len(self.client_folders)} client folders")
            
        except Exception as e:
            logging.error(f"Failed to refresh client list: {str(e)}")
            messagebox.showerror("Error", f"Failed to get client list: {str(e)}")

    def filter_client_list(self, *args):
        """Filter client list based on search text"""
        search_text = self.search_var.get().lower()
        filtered_clients = [
            client for client in self.client_folders 
            if search_text in client.lower()
        ]
        self.update_client_listbox(filtered_clients)

    def update_client_listbox(self, clients: List[str]):
        """Update the client listbox with the provided list"""
        self.client_listbox.delete(0, tk.END)
        for client in clients:
            self.client_listbox.insert(tk.END, client)

    def select_batch(self, count: int):
        """Select the first n clients in the list"""
        self.client_listbox.selection_clear(0, tk.END)
        for i in range(min(count, self.client_listbox.size())):
            self.client_listbox.selection_set(i)

    def clear_selection(self):
        """Clear all selected clients"""
        self.client_listbox.selection_clear(0, tk.END)

    def process_selected_clients(self):
        """Process the selected client folders"""
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
            
        selected_indices = self.client_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one client")
            return
            
        selected_clients = [self.client_listbox.get(i) for i in selected_indices]
        
        # Start processing in a separate thread
        self.processing = True
        threading.Thread(target=self.process_clients, args=(selected_clients,), daemon=True).start()

    def process_clients(self, clients: List[str]):
        """Process the selected client folders in a separate thread"""
        try:
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            self.output_df = pd.DataFrame()
            
            total_clients = len(clients)
            processed_clients = 0
            
            for client in clients:
                if not self.processing:  # Check if processing should stop
                    break
                    
                client_path = self.root_path / client
                self.process_client_folder(client_path)
                
                processed_clients += 1
                progress = (processed_clients / total_clients) * 100
                
                # Update GUI in main thread
                self.window.after(0, self.update_progress, progress, 
                                f"Processing {client} ({processed_clients}/{total_clients})")
            
            if self.processing:  # Only save if not cancelled
                self.save_results()
                
            self.window.after(0, self.processing_complete)
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            self.window.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
        finally:
            self.processing = False

    def update_progress(self, progress_value: float, status_text: str):
        """Update progress bar and status (called from main thread)"""
        self.progress_var.set(progress_value)
        self.progress_label.configure(text=status_text)

    def process_client_folder(self, client_path: Path):
        """Process a single client folder with row limiting and enhanced data filtering."""
        try:
            logging.info(f"Processing client folder: {client_path}")

            excel_files = []
            for file in client_path.glob("**/*.xlsx"):
                if 'wkbk' in file.name.lower():
                    excel_files.append(file)
                    logging.info(f"Found WKBK file: {file}")

            if not excel_files:
                logging.warning(f"No WKBK files found in {client_path}")
                self.results_text.insert(tk.END, f"\nNo WKBK files found in {client_path.name}")
                return

            for file_path in excel_files:
                try:
                    logging.info(f"Attempting to read: {file_path}")
            
                    xls = pd.ExcelFile(file_path)
            
                    logging.info(f"Available sheets in {file_path.name}: {xls.sheet_names}")
            
                    bills_sheet = next((sheet for sheet in xls.sheet_names if sheet.lower() == 'bills'), None)
            
                    if not bills_sheet:
                        logging.warning(f"No 'Bills' sheet found in {file_path.name}")
                        self.results_text.insert(tk.END, f"\nNo 'Bills' sheet in {file_path.name}")
                        continue

                    # Read the entire Bills sheet first
                    df = pd.read_excel(xls, sheet_name=bills_sheet)
            
                    # Find the index where 'Balance' appears in column A
                    balance_idx = df.index[df.iloc[:, 0] == 'Balance'].tolist()
            
                    if balance_idx:
                        end_row = balance_idx[0]
                        logging.info(f"Found 'Balance' at row {end_row + 1}")
                    else:
                        # If 'Balance' not found, use row 21 as a fallback
                        end_row = 21
                        logging.info("'Balance' not found, using default end row 21")

                    # Get column names for debugging
                    logging.info(f"Available columns: {df.columns.tolist()}")
                
                    # Extract columns by letter reference instead of position
                    provider_col = df.columns[0]  # Column A
                    bills_col = df.columns[19] if len(df.columns) > 19 else None  # Column T
                    records_col = df.columns[20] if len(df.columns) > 20 else None  # Column U
                
                    if bills_col is None or records_col is None:
                        # Alternative approach: try to find columns by their position from the end
                        bills_col = df.columns[-2]  # Second to last column
                        records_col = df.columns[-1]  # Last column
                        logging.info(f"Using alternative column selection: Bills={bills_col}, Records={records_col}")

                    # Extract only the relevant columns and rows
                    df_filtered = df.iloc[1:end_row, [0, df.columns.get_loc(bills_col), df.columns.get_loc(records_col)]].copy()
            
                    # Remove any completely empty rows
                    df_filtered = df_filtered[df_filtered.iloc[:, 0].notna()]
            
                    if df_filtered.empty:
                        logging.warning(f"No valid data found in {file_path.name}")
                        continue

                    # Rename columns for clarity
                    df_filtered.columns = ['Provider', 'Has_Bills', 'Has_Records']
            
                    # Clean the provider names (remove any leading/trailing whitespace)
                    df_filtered['Provider'] = df_filtered['Provider'].astype(str).str.strip()
            
                    # Convert checkbox values to Yes/No
                    df_filtered['Has_Bills'] = df_filtered['Has_Bills'].map({True: 'Yes', False: 'No', 1: 'Yes', 0: 'No', 'TRUE': 'Yes', 'False': 'No'})
                    df_filtered['Has_Records'] = df_filtered['Has_Records'].map({True: 'Yes', False: 'No', 1: 'Yes', 0: 'No', 'TRUE': 'Yes', 'False': 'No'})
            
                    # Fill any NaN values in checkbox columns with 'No'
                    df_filtered['Has_Bills'] = df_filtered['Has_Bills'].fillna('No')
                    df_filtered['Has_Records'] = df_filtered['Has_Records'].fillna('No')

                    # Add metadata
                    df_filtered.insert(0, 'Client', client_path.name)
                    df_filtered.insert(1, 'Source_File', file_path.name)
                    df_filtered.insert(2, 'Folder_Path', str(file_path.relative_to(self.root_path)))

                    # Append to main DataFrame
                    self.output_df = pd.concat([self.output_df, df_filtered], ignore_index=True)
            
                    # Update results text
                    summary = f"\nProcessed {file_path.name}:\n"
                    summary += f"Found {len(df_filtered)} providers (rows {3} to {end_row + 1})\n"
                    self.results_text.insert(tk.END, summary)
            
                    # Log the providers found
                    providers_list = df_filtered['Provider'].tolist()
                    logging.info(f"Providers found: {providers_list}")
            
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    self.results_text.insert(tk.END, f"\nError processing {file_path.name}: {str(e)}")

        except Exception as e:
            logging.error(f"Error in client folder {client_path}: {str(e)}")
            self.results_text.insert(tk.END, f"\nError in folder {client_path.name}: {str(e)}")



    def save_results(self):
        """Save processed data to Excel with enhanced formatting."""
        if self.output_df.empty:
            logging.warning("No data to save")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"Client_Analysis_Report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Write main data
                self.output_df.to_excel(writer, sheet_name='Provider_Status', index=False)
                
                # Create summary sheet
                summary_data = {
                    'Metric': [
                        'Total Clients Processed',
                        'Total Files Processed',
                        'Total Providers Found',
                        'Providers with Bills',
                        'Providers with Records'
                    ],
                    'Value': [
                        self.output_df['Client'].nunique(),
                        self.output_df['Source_File'].nunique(),
                        len(self.output_df),
                        len(self.output_df[self.output_df['Has_Bills'] == 'Yes']),
                        len(self.output_df[self.output_df['Has_Records'] == 'Yes'])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
            logging.info(f"Results saved to {output_file}")
            self.results_text.insert(tk.END, f"\nResults saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            self.results_text.insert(tk.END, f"\nError saving results: {str(e)}")

    def processing_complete(self):
        """Display completion message and summary"""
        if self.output_df.empty:
            summary = "No data was processed."
        else:
            summary = f"""Processing Complete!

Total Clients Processed: {self.output_df['Client'].nunique()}
Total Files Processed: {self.output_df['Source_File'].nunique()}
Output File: Client_Analysis_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx

Clients Processed:
{chr(10).join(f"✓ {client}" for client in self.output_df['Client'].unique())}
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, summary)
        self.progress_label.configure(text="Processing Complete")
        
        logging.info("Processing completed successfully")

    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    app = ClientFolderAnalyzer()
    app.run()
