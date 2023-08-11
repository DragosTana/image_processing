	.file	"convolution.cpp"
	.text
	.p2align 4
	.globl	_Z16host_convolutionPKhPKfPhii
	.type	_Z16host_convolutionPKhPKfPhii, @function
_Z16host_convolutionPKhPKfPhii:
.LFB1812:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$32, %rsp
	.cfi_def_cfa_offset 88
	movq	%rdi, (%rsp)
	movq	%rdx, 8(%rsp)
	movl	%ecx, -12(%rsp)
	movl	%r8d, -48(%rsp)
	testl	%r8d, %r8d
	jle	.L1
	testl	%ecx, %ecx
	jle	.L1
	leaq	-1(%rdi), %r10
	pxor	%xmm3, %xmm3
	pxor	%xmm0, %xmm0
	movl	%r8d, %eax
	negl	%eax
	movslq	%r8d, %r8
	movl	$0, -76(%rsp)
	movdqa	.LC0(%rip), %xmm4
	movslq	%eax, %rbx
	movq	%rdi, %rax
	movq	%r8, -56(%rsp)
	addq	%rbx, %rdi
	movq	%rbx, -72(%rsp)
	movq	%rdi, %r11
	movq	%rax, %rdi
	movl	%ecx, %eax
	shrl	$4, %eax
	addq	%r8, %rdi
	subl	$1, %eax
	movq	%rdi, %rbx
	addq	$1, %rax
	salq	$4, %rax
	movq	%rax, -24(%rsp)
	movl	%ecx, %eax
	andl	$-16, %ecx
	leal	-1(%rax), %r14d
	movl	%ecx, %edi
	movl	%ecx, -16(%rsp)
	xorl	%ecx, %ecx
	movl	%r14d, -44(%rsp)
	leaq	36(%rsi), %r14
	movq	%r14, -40(%rsp)
	movslq	%eax, %r14
	movl	%eax, %eax
	movq	%r14, -32(%rsp)
	movq	%rax, 16(%rsp)
	movq	%rdi, -8(%rsp)
	movq	%r8, %rdi
.L4:
	leaq	17(%r11), %rax
	leaq	16(%rdx), %r12
	addl	$1, -76(%rsp)
	leaq	-1(%r11), %r9
	cmpq	%rax, %rdx
	leaq	-1(%rbx), %rbp
	setnb	%r13b
	cmpq	%r9, %r12
	setbe	%al
	orl	%eax, %r13d
	leaq	18(%r10), %rax
	cmpq	%rax, %rdx
	setnb	%al
	cmpq	%r12, %r10
	setnb	%r14b
	orl	%r14d, %eax
	andl	%r13d, %eax
	cmpl	$14, -44(%rsp)
	seta	%r13b
	andl	%r13d, %eax
	leaq	17(%rbx), %r13
	cmpq	%r13, %rdx
	setnb	%r13b
	cmpq	%rbp, %r12
	setbe	%r12b
	orl	%r12d, %r13d
	testb	%r13b, %al
	je	.L12
	movq	-32(%rsp), %rax
	cmpq	-40(%rsp), %rdx
	setnb	%r12b
	addq	%rdx, %rax
	cmpq	%rax, %rsi
	setnb	%al
	orb	%r12b, %al
	je	.L12
	movss	28(%rsi), %xmm2
	movq	%rcx, -64(%rsp)
	leaq	1(%r11), %r15
	xorl	%eax, %eax
	movss	(%rsi), %xmm11
	movss	4(%rsi), %xmm10
	leaq	1(%r10), %r14
	leaq	2(%r10), %r13
	shufps	$0, %xmm2, %xmm2
	movaps	%xmm2, -120(%rsp)
	movss	32(%rsi), %xmm2
	movss	8(%rsi), %xmm9
	movss	12(%rsi), %xmm8
	movss	16(%rsi), %xmm7
	leaq	1(%rbx), %r12
	shufps	$0, %xmm11, %xmm11
	movss	20(%rsi), %xmm6
	movss	24(%rsi), %xmm5
	shufps	$0, %xmm2, %xmm2
	movaps	%xmm2, -104(%rsp)
	movq	-24(%rsp), %rcx
	shufps	$0, %xmm10, %xmm10
	shufps	$0, %xmm9, %xmm9
	shufps	$0, %xmm8, %xmm8
	shufps	$0, %xmm7, %xmm7
	shufps	$0, %xmm6, %xmm6
	shufps	$0, %xmm5, %xmm5
	.p2align 4,,10
	.p2align 3
.L5:
	movdqu	(%r9,%rax), %xmm12
	movdqu	(%rdx,%rax), %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhbw	%xmm3, %xmm12
	punpcklbw	%xmm3, %xmm13
	punpcklbw	%xmm3, %xmm15
	punpckhbw	%xmm3, %xmm14
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm0, %xmm13
	movdqa	%xmm15, %xmm2
	punpcklwd	%xmm0, %xmm1
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm2
	punpckhwd	%xmm0, %xmm15
	mulps	%xmm11, %xmm13
	cvtdq2ps	%xmm1, %xmm1
	cvtdq2ps	%xmm2, %xmm2
	cvtdq2ps	%xmm15, %xmm15
	mulps	%xmm11, %xmm1
	addps	%xmm15, %xmm13
	addps	%xmm2, %xmm1
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm1, %xmm1
	movdqa	%xmm1, %xmm15
	punpcklwd	%xmm13, %xmm1
	punpckhwd	%xmm13, %xmm15
	movdqa	%xmm1, %xmm2
	movdqa	%xmm14, %xmm13
	punpckhwd	%xmm15, %xmm2
	punpcklwd	%xmm15, %xmm1
	punpcklwd	%xmm0, %xmm13
	punpcklwd	%xmm2, %xmm1
	movdqa	%xmm12, %xmm2
	punpckhwd	%xmm0, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm2
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm14
	pand	%xmm4, %xmm1
	mulps	%xmm11, %xmm12
	cvtdq2ps	%xmm2, %xmm2
	cvtdq2ps	%xmm14, %xmm14
	mulps	%xmm11, %xmm2
	addps	%xmm14, %xmm12
	addps	%xmm13, %xmm2
	cvttps2dq	%xmm12, %xmm12
	cvttps2dq	%xmm2, %xmm2
	movdqa	%xmm2, %xmm13
	punpcklwd	%xmm12, %xmm2
	punpckhwd	%xmm12, %xmm13
	movdqa	%xmm2, %xmm12
	punpckhwd	%xmm13, %xmm12
	punpcklwd	%xmm13, %xmm2
	punpcklwd	%xmm12, %xmm2
	pand	%xmm4, %xmm2
	packuswb	%xmm2, %xmm1
	movups	%xmm1, (%rdx,%rax)
	movdqu	(%r11,%rax), %xmm12
	movdqa	%xmm1, %xmm14
	movdqa	%xmm1, %xmm2
	punpcklbw	%xmm3, %xmm14
	punpckhbw	%xmm3, %xmm2
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm1
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm10, %xmm13
	mulps	%xmm10, %xmm1
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm1
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm1, %xmm1
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm1
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm1, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm1
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm1
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm2
	pand	%xmm4, %xmm1
	mulps	%xmm10, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm10, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm2, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm2
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm2, %xmm13
	punpckhwd	%xmm2, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm12, %xmm2
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm2, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm1
	movups	%xmm1, (%rdx,%rax)
	movdqu	(%r15,%rax), %xmm12
	movdqa	%xmm1, %xmm14
	punpckhbw	%xmm3, %xmm1
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm2
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm9, %xmm13
	mulps	%xmm9, %xmm2
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm2
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm2, %xmm2
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm2
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm2, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm2
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm2
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm1
	pand	%xmm4, %xmm2
	mulps	%xmm9, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm9, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm1, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm1
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm1, %xmm13
	punpckhwd	%xmm1, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm12, %xmm1
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm1, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm2
	movups	%xmm2, (%rdx,%rax)
	movdqu	(%r10,%rax), %xmm12
	movdqa	%xmm2, %xmm14
	punpckhbw	%xmm3, %xmm2
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm1
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm8, %xmm13
	mulps	%xmm8, %xmm1
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm1
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm1, %xmm1
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm1
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm1, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm1
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm1
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm2
	pand	%xmm4, %xmm1
	mulps	%xmm8, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm8, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm2, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm2
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm2, %xmm13
	punpckhwd	%xmm2, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm12, %xmm2
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm2, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm1
	movups	%xmm1, (%rdx,%rax)
	movdqu	(%r14,%rax), %xmm12
	movdqa	%xmm1, %xmm14
	punpckhbw	%xmm3, %xmm1
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm2
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm7, %xmm13
	mulps	%xmm7, %xmm2
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm2
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm2, %xmm2
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm2
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm2, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm2
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm2
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm1
	pand	%xmm4, %xmm2
	mulps	%xmm7, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm7, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm1, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm1
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm1, %xmm13
	punpckhwd	%xmm1, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm12, %xmm1
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm1, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm2
	movups	%xmm2, (%rdx,%rax)
	movdqu	0(%r13,%rax), %xmm12
	movdqa	%xmm2, %xmm14
	punpckhbw	%xmm3, %xmm2
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm1
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm6, %xmm13
	mulps	%xmm6, %xmm1
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm1
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm1, %xmm1
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm1
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm1, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm1
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm1
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm2
	pand	%xmm4, %xmm1
	mulps	%xmm6, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm6, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm2, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm2
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm2, %xmm13
	punpckhwd	%xmm2, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm12, %xmm2
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm2, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm1
	movups	%xmm1, (%rdx,%rax)
	movdqu	0(%rbp,%rax), %xmm12
	movdqa	%xmm1, %xmm14
	punpckhbw	%xmm3, %xmm1
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm0, %xmm13
	cvtdq2ps	%xmm15, %xmm15
	punpcklwd	%xmm0, %xmm2
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm2, %xmm2
	mulps	%xmm5, %xmm13
	mulps	%xmm5, %xmm2
	addps	%xmm14, %xmm13
	addps	%xmm15, %xmm2
	cvttps2dq	%xmm13, %xmm13
	cvttps2dq	%xmm2, %xmm2
	movdqa	%xmm2, %xmm14
	punpcklwd	%xmm13, %xmm2
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm2, %xmm13
	punpckhwd	%xmm14, %xmm13
	punpcklwd	%xmm14, %xmm2
	movdqa	%xmm1, %xmm14
	punpcklwd	%xmm13, %xmm2
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpckhwd	%xmm0, %xmm1
	pand	%xmm4, %xmm2
	mulps	%xmm5, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	punpcklwd	%xmm0, %xmm14
	cvtdq2ps	%xmm1, %xmm1
	mulps	%xmm5, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	addps	%xmm1, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm1
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm1, %xmm13
	punpckhwd	%xmm1, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm12, %xmm1
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm1, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm2
	movups	%xmm2, (%rdx,%rax)
	movdqu	(%rbx,%rax), %xmm12
	movdqa	%xmm2, %xmm14
	punpckhbw	%xmm3, %xmm2
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm1
	cvtdq2ps	%xmm15, %xmm15
	punpckhwd	%xmm0, %xmm13
	punpcklwd	%xmm0, %xmm1
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm1, %xmm1
	mulps	-120(%rsp), %xmm1
	addps	%xmm15, %xmm1
	movaps	-120(%rsp), %xmm15
	mulps	%xmm15, %xmm13
	cvttps2dq	%xmm1, %xmm1
	addps	%xmm14, %xmm13
	movdqa	%xmm1, %xmm14
	cvttps2dq	%xmm13, %xmm13
	punpcklwd	%xmm13, %xmm1
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm1, %xmm13
	punpcklwd	%xmm14, %xmm1
	punpckhwd	%xmm14, %xmm13
	movdqa	%xmm2, %xmm14
	punpckhwd	%xmm0, %xmm2
	punpcklwd	%xmm13, %xmm1
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	cvtdq2ps	%xmm2, %xmm2
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpcklwd	%xmm0, %xmm14
	pand	%xmm4, %xmm1
	mulps	%xmm15, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	mulps	%xmm15, %xmm13
	addps	%xmm2, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm2
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm2, %xmm13
	punpckhwd	%xmm2, %xmm12
	movdqa	%xmm13, %xmm2
	punpckhwd	%xmm12, %xmm2
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm2, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm1
	movups	%xmm1, (%rdx,%rax)
	movdqu	(%r12,%rax), %xmm12
	movdqa	%xmm1, %xmm14
	punpckhbw	%xmm3, %xmm1
	punpcklbw	%xmm3, %xmm14
	movdqa	%xmm12, %xmm13
	movdqa	%xmm14, %xmm15
	punpckhwd	%xmm0, %xmm14
	punpcklbw	%xmm3, %xmm13
	punpcklwd	%xmm0, %xmm15
	cvtdq2ps	%xmm14, %xmm14
	punpckhbw	%xmm3, %xmm12
	movdqa	%xmm13, %xmm2
	cvtdq2ps	%xmm15, %xmm15
	punpckhwd	%xmm0, %xmm13
	punpcklwd	%xmm0, %xmm2
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm2, %xmm2
	mulps	-104(%rsp), %xmm2
	addps	%xmm15, %xmm2
	movaps	-104(%rsp), %xmm15
	mulps	%xmm15, %xmm13
	cvttps2dq	%xmm2, %xmm2
	addps	%xmm14, %xmm13
	movdqa	%xmm2, %xmm14
	cvttps2dq	%xmm13, %xmm13
	punpcklwd	%xmm13, %xmm2
	punpckhwd	%xmm13, %xmm14
	movdqa	%xmm2, %xmm13
	punpcklwd	%xmm14, %xmm2
	punpckhwd	%xmm14, %xmm13
	movdqa	%xmm1, %xmm14
	punpckhwd	%xmm0, %xmm1
	punpcklwd	%xmm13, %xmm2
	movdqa	%xmm12, %xmm13
	punpckhwd	%xmm0, %xmm12
	cvtdq2ps	%xmm1, %xmm1
	punpcklwd	%xmm0, %xmm13
	cvtdq2ps	%xmm12, %xmm12
	punpcklwd	%xmm0, %xmm14
	pand	%xmm4, %xmm2
	mulps	%xmm15, %xmm12
	cvtdq2ps	%xmm13, %xmm13
	cvtdq2ps	%xmm14, %xmm14
	mulps	%xmm15, %xmm13
	addps	%xmm1, %xmm12
	addps	%xmm14, %xmm13
	cvttps2dq	%xmm12, %xmm1
	cvttps2dq	%xmm13, %xmm13
	movdqa	%xmm13, %xmm12
	punpcklwd	%xmm1, %xmm13
	punpckhwd	%xmm1, %xmm12
	movdqa	%xmm13, %xmm1
	punpckhwd	%xmm12, %xmm1
	punpcklwd	%xmm12, %xmm13
	punpcklwd	%xmm1, %xmm13
	pand	%xmm4, %xmm13
	packuswb	%xmm13, %xmm2
	movups	%xmm2, (%rdx,%rax)
	addq	$16, %rax
	cmpq	%rcx, %rax
	jne	.L5
	movl	-12(%rsp), %r14d
	movl	-16(%rsp), %r15d
	movq	-64(%rsp), %rcx
	cmpl	%r15d, %r14d
	je	.L9
	movq	-8(%rsp), %r9
	movq	-72(%rsp), %r12
	movq	%r10, -120(%rsp)
	leaq	(%r9,%rcx), %rax
	addq	%r12, %r9
	addq	(%rsp), %r9
	addq	8(%rsp), %rax
	subl	%r9d, %r15d
	movl	%r15d, %r13d
	.p2align 4,,10
	.p2align 3
.L7:
	movzbl	-1(%r9), %r10d
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	addq	$1, %rax
	cvtsi2ssl	%r10d, %xmm1
	movzbl	-1(%rax), %r10d
	mulss	(%rsi), %xmm1
	cvtsi2ssl	%r10d, %xmm2
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r10d
	pxor	%xmm1, %xmm1
	movb	%r10b, -1(%rax)
	movzbl	(%r9), %ebp
	movzbl	%r10b, %r10d
	cvtsi2ssl	%r10d, %xmm2
	cvtsi2ssl	%ebp, %xmm1
	mulss	4(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r10d
	pxor	%xmm1, %xmm1
	movb	%r10b, -1(%rax)
	movzbl	1(%r9), %ebp
	movzbl	%r10b, %r10d
	cvtsi2ssl	%r10d, %xmm2
	movq	%r9, %r10
	cvtsi2ssl	%ebp, %xmm1
	mulss	8(%rsi), %xmm1
	subq	%r12, %r10
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	-1(%r10,%rcx), %r15d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r15d, %xmm1
	mulss	12(%rsi), %xmm1
	leaq	(%r9,%rcx), %r15
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	(%r15,%r8), %r15d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r15d, %xmm1
	mulss	16(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	1(%r10,%rcx), %r15d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r15d, %xmm1
	mulss	20(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	-1(%r10,%rdi), %r15d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r15d, %xmm1
	mulss	24(%rsi), %xmm1
	leaq	(%r9,%rdi), %r15
	addq	$1, %r9
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	(%r15,%r8), %r15d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r15d, %xmm1
	mulss	28(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %ebp
	pxor	%xmm1, %xmm1
	movb	%bpl, -1(%rax)
	movzbl	1(%r10,%rdi), %r10d
	movzbl	%bpl, %ebp
	cvtsi2ssl	%ebp, %xmm2
	cvtsi2ssl	%r10d, %xmm1
	mulss	32(%rsi), %xmm1
	addss	%xmm2, %xmm1
	cvttss2sil	%xmm1, %r10d
	movb	%r10b, -1(%rax)
	leal	0(%r13,%r9), %r10d
	cmpl	%r10d, %r14d
	jg	.L7
	movq	-120(%rsp), %r10
.L9:
	movq	-56(%rsp), %rax
	movl	-76(%rsp), %r14d
	addq	%rax, -72(%rsp)
	addq	%rax, %r10
	addq	%rax, %rdx
	addq	%rax, %r11
	addq	%rax, %rbx
	addq	%rax, %rcx
	subq	%rax, %r8
	addq	%rax, %rdi
	cmpl	%r14d, -48(%rsp)
	jne	.L4
.L1:
	addq	$32, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L12:
	.cfi_restore_state
	movq	16(%rsp), %r14
	leaq	1(%r10), %r12
	movq	%rbx, %rbp
	movq	%r11, %r9
	movq	%rdx, %rax
	leaq	(%r14,%r11), %r15
	.p2align 4,,10
	.p2align 3
.L8:
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	addq	$1, %r9
	addq	$1, %rax
	movzbl	-2(%r9), %r13d
	addq	$1, %r12
	addq	$1, %rbp
	cvtsi2ssl	%r13d, %xmm1
	movzbl	-1(%rax), %r13d
	mulss	(%rsi), %xmm1
	cvtsi2ssl	%r13d, %xmm2
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	-1(%r9), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	4(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	(%r9), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	8(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	-2(%r12), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	12(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	-1(%r12), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	16(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	(%r12), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	20(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	-2(%rbp), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	24(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	-1(%rbp), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	28(%rsi), %xmm1
	addss	%xmm2, %xmm1
	pxor	%xmm2, %xmm2
	cvttss2sil	%xmm1, %r13d
	pxor	%xmm1, %xmm1
	movb	%r13b, -1(%rax)
	movzbl	0(%rbp), %r14d
	movzbl	%r13b, %r13d
	cvtsi2ssl	%r13d, %xmm2
	cvtsi2ssl	%r14d, %xmm1
	mulss	32(%rsi), %xmm1
	addss	%xmm2, %xmm1
	cvttss2sil	%xmm1, %r13d
	movb	%r13b, -1(%rax)
	cmpq	%r15, %r9
	jne	.L8
	jmp	.L9
	.cfi_endproc
.LFE1812:
	.size	_Z16host_convolutionPKhPKfPhii, .-_Z16host_convolutionPKhPKfPhii
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I__Z16host_convolutionPKhPKfPhii, @function
_GLOBAL__sub_I__Z16host_convolutionPKhPKfPhii:
.LFB2294:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	popq	%rbp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE2294:
	.size	_GLOBAL__sub_I__Z16host_convolutionPKhPKfPhii, .-_GLOBAL__sub_I__Z16host_convolutionPKhPKfPhii
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z16host_convolutionPKhPKfPhii
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.value	255
	.value	255
	.value	255
	.value	255
	.value	255
	.value	255
	.value	255
	.value	255
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
