# hw14.py

print('6---------------')
import re
text ="이메일 목록: test@example.com, hello@world.net, user123@domain.org"
pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%+-]+\.[a-zA-Z0-9._%+-]{2,}'
emails = re.findall(pattern, text)
print(emails)

print('7---------------')
text = "연락처: 010-1234-5678, 02-987-6543, 031-456-7890"
pattern = r'\d+-+\d+-+\d{2,}'
phone_number = re.findall(pattern, text)
print(phone_number)

print('8---------------')
text = "I love Python. Java is also popular. Python is great for AI."
pattern = r'[^.]*Python[^.]*\.'
python= re.findall(pattern,text)
print(python)

print('9---------------')
text = "상품 코드: A123, B456, C789, 가격: 12000원"
pattern = r'[\d+}]'
nums = re.findall(pattern,text)
nums=list(map(int,nums))
print(nums)

print('10--------------')
text = "NASA is working on AI projects with IBM and Google."
pattern = r'[A-Z]{2,}'
large=re.findall(pattern,text)
print(large)