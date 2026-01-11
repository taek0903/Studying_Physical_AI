import tkinter as tk
from tkinter import ttk, messagebox

class PizzaOrderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("조각 피자 주문 프로그램")
        self.root.geometry("420x420")

        # === 가격 설정 ===
        self.base_prices = {
            "치즈": 3500,
            "페퍼로니": 4000,
            "하와이안": 4200,
            "불고기": 4500
        }

        self.topping_prices = {
            "치즈 추가": 500,
            "올리브": 300,
            "베이컨": 700,
            "파인애플": 500
        }

        # === 선택 값 변수 ===
        self.selected_pizza = tk.StringVar(value="치즈")
        self.topping_vars = {}
        self.slice_count = tk.IntVar(value=1)
        self.total_price = tk.IntVar(value=0)

        self.create_widgets()
        self.update_price()  # 시작할 때 한 번 계산

    def create_widgets(self):
        # ===== 피자 종류 선택 =====
        frame_pizza = ttk.LabelFrame(self.root, text="피자 종류 선택")
        frame_pizza.pack(fill="x", padx=10, pady=10)

        for name in self.base_prices:
            ttk.Radiobutton(
                frame_pizza,
                text=f"{name} ({self.base_prices[name]}원/조각)",
                value=name,
                variable=self.selected_pizza,
                command=self.update_price
            ).pack(anchor="w", padx=5, pady=2)

        # ===== 토핑 선택 =====
        frame_topping = ttk.LabelFrame(self.root, text="추가 토핑 (조각당)")
        frame_topping.pack(fill="x", padx=10, pady=10)

        for name, price in self.topping_prices.items():
            var = tk.BooleanVar(value=False)
            self.topping_vars[name] = var
            ttk.Checkbutton(
                frame_topping,
                text=f"{name} (+{price}원)",
                variable=var,
                command=self.update_price
            ).pack(anchor="w", padx=5, pady=2)

        # ===== 수량 입력 =====
        frame_count = ttk.LabelFrame(self.root, text="주문 수량")
        frame_count.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_count, text="조각 수:").pack(side="left", padx=5, pady=5)

        spin = ttk.Spinbox(
            frame_count,
            from_=1,
            to=20,
            textvariable=self.slice_count,
            width=5,
            command=self.update_price,
            justify="center"
        )
        spin.pack(side="left", padx=5, pady=5)
        spin.bind("<KeyRelease>", lambda e: self.update_price())

        # ===== 결제 정보 (주문 금액 표시) =====
        frame_price = ttk.LabelFrame(self.root, text="결제 정보")
        frame_price.pack(fill="x", padx=10, pady=10)

        # 👉 여기서 주문 금액이 항상 보이게!
        self.label_price = ttk.Label(
            frame_price,
            text="주문 금액: 0원",
            font=("맑은 고딕", 12, "bold")
        )
        self.label_price.pack(anchor="center", pady=5)

        # ===== 버튼 영역 (가격 계산 + 주문하기) =====
        frame_buttons = ttk.Frame(self.root)
        frame_buttons.pack(fill="x", padx=10, pady=10)

        # 가격 계산 버튼
        btn_calc = ttk.Button(frame_buttons, text="가격 계산", command=self.update_price)
        btn_calc.pack(side="left", expand=True, fill="x", padx=5)

        # 👉 여기 “주문하기” 버튼 추가
        btn_order = ttk.Button(frame_buttons, text="주문하기", command=self.confirm_order)
        btn_order.pack(side="left", expand=True, fill="x", padx=5)

    # ===== 로직 부분 =====
    def calculate_total(self):
        # 기본 피자 가격
        base_price = self.base_prices[self.selected_pizza.get()]

        # 선택된 토핑 가격 합 (조각당)
        topping_total = 0
        for name, var in self.topping_vars.items():
            if var.get():
                topping_total += self.topping_prices[name]

        # 조각 수
        try:
            count = int(self.slice_count.get())
            if count < 1:
                count = 1
                self.slice_count.set(1)
        except ValueError:
            count = 1
            self.slice_count.set(1)

        # 총액 = (기본 + 토핑) * 조각 수
        total = (base_price + topping_total) * count
        return total, count, topping_total

    def update_price(self):
        total, _, _ = self.calculate_total()
        self.total_price.set(total)
        # 👉 레이블에 주문 금액 출력
        self.label_price.config(text=f"주문 금액: {total:,}원")

    def confirm_order(self):
        total, count, topping_total = self.calculate_total()

        # 선택된 토핑 문자열 만들기
        selected_toppings = [name for name, var in self.topping_vars.items() if var.get()]
        if selected_toppings:
            toppings_text = ", ".join(selected_toppings)
        else:
            toppings_text = "추가 토핑 없음"

        msg = (
            f"주문 내역을 확인해 주세요.\n\n"
            f"- 피자 종류: {self.selected_pizza.get()}\n"
            f"- 조각 수: {count} 조각\n"
            f"- 토핑: {toppings_text}\n"
            f"- 조각당 추가 토핑 금액: {topping_total}원\n\n"
            f"최종 주문 금액은 {total:,}원 입니다."
        )

        messagebox.showinfo("주문 완료", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = PizzaOrderApp(root)
    root.mainloop()
