use64
format ELF64

public find_index


section '.rodata' align 16

compare: times 16 db '-'
one: times 16 db 1


section '.text' executable align 16

find_index:
movapd  xmm8,[compare]
movapd  xmm9,[one]
xorps   xmm10,xmm10
xor     rdx,rdx
mov     r8,rdi

align 16

find_index_loop:
movaps  xmm0,[rdi+0x00]
movaps  xmm1,[rdi+0x10]
movaps  xmm2,[rdi+0x20]
movaps  xmm3,[rdi+0x30]
movaps  xmm4,[rdi+0x40]
movaps  xmm5,[rdi+0x50]
movaps  xmm6,[rdi+0x60]
movaps  xmm7,[rdi+0x70]

pcmpeqb xmm0,xmm8
pcmpeqb xmm1,xmm8
pcmpeqb xmm2,xmm8
pcmpeqb xmm3,xmm8
pcmpeqb xmm4,xmm8
pcmpeqb xmm5,xmm8
pcmpeqb xmm6,xmm8
pcmpeqb xmm7,xmm8

paddb   xmm0,xmm9
paddb   xmm1,xmm9
paddb   xmm2,xmm9
paddb   xmm3,xmm9
paddb   xmm4,xmm9
paddb   xmm5,xmm9
paddb   xmm6,xmm9
paddb   xmm7,xmm9

paddb   xmm0,xmm1
paddb   xmm2,xmm3
paddb   xmm4,xmm5
paddb   xmm6,xmm7
paddb   xmm0,xmm2
paddb   xmm4,xmm6
paddb   xmm0,xmm4

add     rdi,0x80

psadbw  xmm0,xmm10
movd    eax,xmm0
psrldq  xmm0,8
movd    edx,xmm0
add     eax,edx

sub     rsi,rax
ja      find_index_loop

je      done

add     rsi,rax
sub     rdi,0x80

single_loop:
mov     al,[rdi]
add     rdi,1
cmp     al,'-'
je      single_loop
sub     rsi,1
ja      single_loop

done:

mov     rax,rdi
sub     rax,r8

ret
