data segment
    error_input db 'Input error$'
data ends

code segment
assume cs:code, ds:data

start:
    mov  ax, data
    mov  ds, ax

    mov  ah, 01h
    int  21h

    cmp  al, '0'
    jb   Error_output
    cmp  al, '9'
    ja   Error_output

    sub  al, '0'           
    mov  dl, al
    mov  bl, dl             
    mul  dl      ;al=al*dl              
    mov  cl, al             
    mov  al, bl             
    mul  cl      ;al=al*cl           
    mov  dl, al            


    mov  ax, 4C00h
    int  21h

Error_output:

    mov  ah, 09h
    lea  dx, error_input
    int  21h

    mov  ax, 4C00h
    int  21h

code ends
end start
