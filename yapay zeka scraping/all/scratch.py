import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv

class YouTubeCommentViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Yorumları Görüntüleyici")
        self.root.geometry("800x600")

        # CSV Dosyasını Seçme
        self.load_button = tk.Button(self.root, text="CSV Dosyasını Yükle", command=self.load_csv)
        self.load_button.pack(pady=20)

        # Video Başlıkları Seçimi
        self.video_list_label = tk.Label(self.root, text="Video Başlığı Seçin:")
        self.video_list_label.pack(pady=10)

        self.video_combobox = ttk.Combobox(self.root, state="readonly", width=50)
        self.video_combobox.pack(pady=10)

        self.select_button = tk.Button(self.root, text="Yorumları Göster", command=self.show_comments)
        self.select_button.pack(pady=20)

        # Yorumların Görüntüleneceği Alan
        self.result_text = tk.Text(self.root, width=100, height=20)
        self.result_text.pack(pady=20)

        self.csv_data = []  # Veriler buraya yüklenecek

    def load_csv(self):
        # Dosya seçme penceresi
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.read_csv(file_path)
        else:
            messagebox.showwarning("Hata", "Geçerli bir CSV dosyası seçmediniz!")

    def read_csv(self, file_path):
        try:
            # CSV dosyasını oku
            with open(file_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                header = next(reader)  # Başlıkları atla
                self.csv_data = [row for row in reader]

            # Video başlıklarını combobox'a ekle
            video_titles = list(set(row[0] for row in self.csv_data))  # Video başlıklarını al, tekrarsız
            self.video_combobox['values'] = video_titles

            if not video_titles:
                messagebox.showinfo("Bilgi", "CSV dosyasında video başlığı bulunamadı.")
        except Exception as e:
            messagebox.showerror("Hata", f"CSV dosyası okunurken bir hata oluştu: {str(e)}")

    def show_comments(self):
        selected_video = self.video_combobox.get()
        if not selected_video:
            messagebox.showwarning("Seçim Hatası", "Bir video başlığı seçmelisiniz.")
            return

        # Seçilen video başlığına ait yorumları filtrele
        selected_comments = [row for row in self.csv_data if row[0] == selected_video]

        # Önceki içerikleri temizle
        self.result_text.delete(1.0, tk.END)

        if selected_comments:
            # Yorumları yazdır
            for row in selected_comments:
                video_title = row[0]
                user_name = row[1]
                comment_date = row[2]
                comment_text = row[3]
                likes = row[4]
                reply_count = row[5]
                Predicted_Class=row[6]

                self.result_text.insert(tk.END, f"Video Başlığı: {video_title}\n")
                self.result_text.insert(tk.END, f"Yorum Yapan: {user_name}\n")
                self.result_text.insert(tk.END, f"Tarih: {comment_date}\n")
                self.result_text.insert(tk.END, f"Yorum: {comment_text}\n")
                self.result_text.insert(tk.END, f"Beğeni Sayısı: {likes}\n")
                self.result_text.insert(tk.END, f"Yanıt Sayısı: {reply_count}\n")
                self.result_text.insert(tk.END, f"Yorum Durumu: {Predicted_Class}\n")

                self.result_text.insert(tk.END, "-" * 60 + "\n")
        else:
            self.result_text.insert(tk.END, "Bu video için yorum bulunamadı.\n")

# Uygulama Başlatma
root = tk.Tk()
app = YouTubeCommentViewerApp(root)
root.mainloop()
