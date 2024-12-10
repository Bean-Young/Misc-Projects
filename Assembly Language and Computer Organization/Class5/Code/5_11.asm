Code  segment
    assume   cs:code

savereg   macro   ; 入栈保存
    push  ax
    push  bx
    push  cx
    push  dx
  endm

restorereg   macro   ;出栈恢复
    pop  dx
    pop  cx
    pop  bx
    pop  ax
  endm

start:   mov  ax,1
         mov  bx,2
        savereg   ; 保存寄存器值
        add  ax, 10  ; 变动 ax, bx 的值
        sub  bx, 5
        restorereg  ; 恢复寄存器。应该可见(ax)=1, (bx)=2 恢复初值
mov  ah, 4ch
        int   21h
CODE  	ENDS
	END	START