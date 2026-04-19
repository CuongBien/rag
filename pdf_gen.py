# Sample PDF generator (fpdf). Install: pip install fpdf
from fpdf import FPDF


class PDF(FPDF):
    def header(self) -> None:
        self.set_font("Arial", "B", 12)
        # cell(w, h, text, border, ln, align): full-width title row, centered.
        self.cell(0, 10, "Bao cao Chien luoc Tap doan Alpha 2024", 0, 1, "C")


pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=11)

content = [
    "1. Tong quan to chuc:",
    "Tap doan Alpha (Alpha Group) co tru so tai Silicon Valley, duoc dieu hanh boi CEO Ly Hoang Nam.",
    "Alpha Group so huu hai cong ty con la Beta Logistics va Gamma AI.",
    "",
    "2. Moi quan he doi tac:",
    "Vao thang 5/2024, Beta Logistics da ky hop dong voi OmniStore.",
    "Nguoi dai dien phia OmniStore la ba Elena Rodriguez.",
    "",
    "3. Du an phat trien:",
    "Gamma AI dang phat trien Project-X. Alpha Group da mua cong nghe tu DeepSensors (Berlin).",
    "Ong Ly Hoang Nam truc tiep giam sat du an nay."
]

for line in content:
    # cell width 200 (page units), height 10, newline after each line.
    pdf.cell(200, 10, txt=line, ln=True)

pdf.output("strategic_report.pdf")  # output path for the generated PDF file
print("Da tao file strategic_report.pdf thanh cong!")