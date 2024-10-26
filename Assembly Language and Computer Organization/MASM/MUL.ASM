assume cs:p1,ds:data
data segment
  info db '2*3='
  result db 0,'$'
data ends
p1 segment
start:
  mov ax,data
  mov ds,ax
  mov al,2
  mov bl,3
  mul bl
  add al,30h
  mov result[0],al

  mov dx,offset info
  mov ax,0900h
  int 21h
  mov ax,4c00h
  int 21h
p1 ends
  end start
