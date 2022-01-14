import socket
import multiprocessing
import threading
import sys
import select
import os
import queue
import time
import torch.nn as nn
import torch
import numpy as np
import pickle
 


SERVER_ADDRESS='./pip1.socketpipe'
MODE = 1 #1 for server and 0 for client

def recv_msg(udp_socket,que,stop):
    while True:
        # recv msg
        recv_data = udp_socket.recv(1024*1024)
        # print("%s : %s" % (str(recv_data[1]), recv_data[0].decode('utf-8')))
        que.put(recv_data)
        if not stop.empty():
            return
 
 
def send_msg(udp_socket,que,stop):
    while True:
        # send msg
        if not que.empty():
            send_data = que.get()
        # udp_socket.sendto(send_data.encode('utf-8'), (dest_ip, dest_port))
            udp_socket.sendall(send_data)
        if not stop.empty():
            return
 
 
def socket_ini():
 
    if MODE == 1:
        try:
            os.unlink(SERVER_ADDRESS)
        except OSError:
            if os.path.exists(SERVER_ADDRESS):
                raise
    udp_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    my_socket=udp_socket

    if MODE == 1:
        udp_socket.bind(SERVER_ADDRESS)
        udp_socket.listen(1)
        print ('waiting for a connection')
        connection, client_address = udp_socket.accept()
        my_socket=connection
        print('connected!')
    elif MODE ==0:
        try:
            udp_socket.connect(SERVER_ADDRESS)
            print('connected successfully')
        except socket.error:
            print ("socket error")
            sys.exit(1)
    else:
        raise


    read_queue=queue.Queue()
    write_queue=queue.Queue()
    stop_queue=queue.Queue()
    thread_recv = threading.Thread(target=recv_msg, args=(my_socket,read_queue,stop_queue))
    thread_send = threading.Thread(target=send_msg, args=(my_socket,write_queue,stop_queue))
    thread_recv.start()
    thread_send.start()
    return thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket
 
def socket_del(thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket):
    stop_queue.put("stop")
    my_socket.sendall(pickle.dumps("over"))
    thread_recv.join()
    thread_send.join()
    if MODE==1:
        my_socket.close()
    udp_socket.close() 

    pass

def test(read_queue,write_queue):
    while True:
        if not read_queue.empty():
            s = read_queue.get()
            print('<<<<', s)
            if s=='quit'or s==b'quit':
                write_queue.put('quit'.encode())
                break

        if select.select([sys.stdin], [], [], 0.0)[0]:
            info=input().encode()
            write_queue.put(info)
            if info=='quit' or info==b'quit':
                read_queue.put('quit'.encode())
                break
    pass


def main():
    thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket=socket_ini()
    test(read_queue,write_queue)
    socket_del(thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket)
    #test(read_queue,write_queue)
 
 
if __name__ == '__main__':
    main()
