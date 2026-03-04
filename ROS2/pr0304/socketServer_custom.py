import socket

# TCP/IP 소켓 생성 (IPv4, TCP)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버 소켓을 특정 IP와 포트에 바인딩
server_socket.bind(("127.0.0.1", 12345))

# 클라이언트의 연결 요청을 기다리도록 설정 (동시에 1개의 연결까지 대기)
server_socket.listen(1)
print("서버가 시작되었습니다. 클라이언트를 기다리는 중...")

# 클라이언트의 연결 요청 수락
connection, address = server_socket.accept()
print(f"클라이언트가 연결되었습니다: {address}")

# 클라이언트가 보낸 데이터 수신 (최대 1024바이트)
data = connection.recv(1024)
print(f"클라이언트로부터 받은 데이터: {data.decode()}")

# 받은 데이터를 그대로 클라이언트에게 다시 전송 (에코 서버 기능)
connection.sendall(data)

# 연결 종료
connection.close()
server_socket.close()
