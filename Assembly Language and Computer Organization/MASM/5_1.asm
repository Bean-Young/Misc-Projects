DATA SEGMENT
    ; (1) 定义一个数组，类型为字节，其中存放"ABCDEFGH"
    array1 DB 'ABCDEFGH$'

    ; (2) 定义一个字节区域，第一个字节为10，其后连续存放10个初值为0的连续字节。
    byte_region DB 10 DUP(0)

    ; (3) 将'byte', 'word'存在某一数据区
    char DB 'byte$'  
    d_word DB 'word$'   

DATA ENDS

CODE SEGMENT
    ASSUME CS:CODE, DS:DATA
START:
	MOV AH, 4CH
    INT 21H
CODE ENDS

END START
