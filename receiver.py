#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import socket
import pandas as pd
import numpy as np
import sys
import selectors

HOST = ''
PORT = 9700


def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    address = (HOST, PORT)

    print("Socket is created!")
    try:
        s.bind(address)
    except socket.error as err:
        print("Binding Failed, Error code is:" + str(err[0]) + ", Message:" + err[1])
        sys.exit()
    print("Socket binded successfully!")

    s.listen(1)
    print("Socket is now listening!")

    return s


def cleanString(incomingString):
    newstring = incomingString
    newstring = newstring.replace("\n", "")
    return newstring


def callback(data, main_mat, first_part, second_part):
    cleaned_str = cleanString(data)
    first_part = [list(l) for l in zip(*first_part.values)]
    second_part = [list(l) for l in zip(*second_part.values)]
    for item, val in enumerate(first_part[0]):
        if cleaned_str == val:
            res = main_mat.iloc[:, item]
            break
    return res.iloc[res.nonzero()[0]]


def main():
    socket_list = []
    main_mat = np.genfromtxt('final_dict.csv', delimiter=',')
    first_part = pd.read_csv('first_part.csv', index_col=None, header=None)
    second_part = pd.read_csv('second_part.csv', index_col=None, header=None)
    main_mat = pd.DataFrame(main_mat, index=second_part, columns=first_part)
    s = connect()
    socket_list.append(s)

    while True:
        conn, addr = s.accept()
        socket_list.append(conn)
        print("Connected with " + addr[0] + ":" + str(addr[1]))
        data = conn.recv(1024)
        conn.setblocking(False)
        response_array = callback(data.decode('UTF-8'), main_mat, first_part, second_part)
        temp = response_array.index.values
        final_res = []
        for i in range(len(temp)):
            final_res.append(temp[i][0])
        conn.send(pickle.dumps(final_res))
        print("The response to client is:")
        print(final_res)


if __name__ == '__main__':
    main()
