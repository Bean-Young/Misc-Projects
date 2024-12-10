data segment
  BUF1  DW  -5, 2, 4, -10, 9, -8, 10, 20
  BUF2  DW  4, -2, 24, 45, -25, 20, 30, 10
  S     DW  8 DUP (0)
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax
    mov  es, ax

    mov  cx, 8
    mov  si, OFFSET BUF1
    mov  di, OFFSET BUF2
    mov  bx, OFFSET S

sum_loop:
    mov  ax, [si]
    add  ax, [di]
    mov  [bx], ax

    add  si, 2
    add  di, 2
    add  bx, 2
    loop sum_loop

    mov  ax, 4C00h
    int  21h

code ends
end start
