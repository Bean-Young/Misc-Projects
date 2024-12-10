Code  segment
Assume  cs:code
 Mov  ax,0
 Mov  ds,ax       ;设置ds段寄存器
 Mov  ax, 0200h
 Mov  es, ax      ; 设置es附加段寄存器
 Mov  bx,0        ;初始的偏移地址
 Mov  cx,128      ;循环次数
S: mov  al, [bx]
  Mov  es:[bx], al   ; 使用es 段前缀
  Inc  bx          ; 偏移地址递增1
  Loop  s          ; 循环 
;执行完循环后，在debug中观察上述两个数据段内容是否一致
  int 3
  Mov ax,  4c00h
  Int  21h 
 Code  ends
   end