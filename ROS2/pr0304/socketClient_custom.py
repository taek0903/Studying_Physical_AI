import socket

# TCP/IP 소켓 생성 (IPv4, TCP)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip_adress = input('접속할 서버의 IP를 입력하세요: ')
port_number = int(input('접속할 서버의 포트를 입력하세요: '))

# 서버에 연결 (IP: 127.0.0.1, 포트: 12345)
client_socket.connect((ip_adress, port_number))

print("서버에 연결되었습니다.")

# 서버로 데이터 전송
client_socket.sendall(b'Hello, Server!')

# 서버로부터 최대 1024바이트 데이터 수신
data = client_socket.recv(1024)
print(f"서버로부터 받은 데이터: {data.decode()}")

# 소켓 연결 종료
client_socket.close()