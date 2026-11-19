// SPDX-License-Identifier: GPL-2.0
/*
 * eBPF probes: execve, ptrace, commit_creds, network sockets, fork/exit.
 * CO-RE without vmlinux.h: only the few kernel fields we need are declared
 * (preserve_access_index lets libbpf relocate them against the running kernel's BTF).
 */
#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>
#include "events.h"

#pragma clang attribute push(__attribute__((preserve_access_index)), apply_to = record)
typedef struct { __u32 val; } kuid_t;
struct cred {
	kuid_t uid;
	kuid_t euid;
};
struct task_struct {
	int pid;
	int tgid;
	struct task_struct *real_parent;
	const struct cred *cred;
};
#pragma clang attribute pop

struct sys_enter_ctx {
	__u64 common;
	long id;
	unsigned long args[6];
};

struct sched_fork_ctx {
	__u64 common;
	char parent_comm[16];
	int parent_pid;
	char child_comm[16];
	int child_pid;
};

struct sched_exit_ctx {
	__u64 common;
	char comm[16];
	int pid;
	int prio;
};

struct sockaddr_in_min {
	__u16 family;
	__u16 port; /* network order */
	__u32 addr; /* network order */
};

#define AF_INET 2
#define AF_INET6 10
#define MAX_ENVP 24

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 20);
} events SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);
} drops SEC(".maps");

static __always_inline struct event *event_begin(__u32 type)
{
	struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	if (!e) {
		__u32 k = 0;
		__u64 *d = bpf_map_lookup_elem(&drops, &k);
		if (d)
			(*d)++;
		return 0;
	}
	struct task_struct *t = (struct task_struct *)bpf_get_current_task();
	__u64 pt = bpf_get_current_pid_tgid();
	__u64 ug = bpf_get_current_uid_gid();

	e->ts_ns = bpf_ktime_get_ns();
	e->pid = pt >> 32;
	e->tid = (__u32)pt;
	e->uid = (__u32)ug;
	e->euid = BPF_CORE_READ(t, cred, euid.val);
	e->ppid = BPF_CORE_READ(t, real_parent, tgid);
	e->type = type;
	e->arg0 = e->arg1 = e->arg2 = 0;
	e->flags = 0;
	e->pad = 0;
	e->data[0] = 0;
	bpf_get_current_comm(&e->comm, sizeof(e->comm));
	return e;
}

static __always_inline int has_ld_preload(const char *const *envp)
{
	char buf[12];
#pragma unroll
	for (int i = 0; i < MAX_ENVP; i++) {
		const char *p = 0;
		if (bpf_probe_read_user(&p, sizeof(p), &envp[i]) || !p)
			return 0;
		if (bpf_probe_read_user(buf, sizeof(buf), p))
			continue;
		if (buf[0] == 'L' && buf[1] == 'D' && buf[2] == '_' && buf[3] == 'P' && buf[4] == 'R' &&
		    buf[5] == 'E' && buf[6] == 'L' && buf[7] == 'O' && buf[8] == 'A' && buf[9] == 'D' &&
		    buf[10] == '=')
			return 1;
	}
	return 0;
}

SEC("tp/syscalls/sys_enter_execve")
int tp_execve(struct sys_enter_ctx *ctx)
{
	struct event *e = event_begin(EVT_EXECVE);
	if (!e)
		return 0;
	bpf_probe_read_user_str(e->data, sizeof(e->data), (const void *)ctx->args[0]);
	if (has_ld_preload((const char *const *)ctx->args[2]))
		e->flags |= FLAG_LD_PRELOAD_ENV;
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/syscalls/sys_enter_ptrace")
int tp_ptrace(struct sys_enter_ctx *ctx)
{
	struct event *e = event_begin(EVT_PTRACE);
	if (!e)
		return 0;
	e->arg0 = ctx->args[0]; /* request */
	e->arg1 = ctx->args[1]; /* target pid */
	bpf_ringbuf_submit(e, 0);
	return 0;
}

/* Called for every credential commit (setuid(), execve of setuid binaries, kernel exploits). */
SEC("kprobe/commit_creds")
int kp_commit_creds(struct pt_regs *ctx)
{
	const struct cred *newc = (const struct cred *)PT_REGS_PARM1(ctx);
	struct event *e = event_begin(EVT_COMMIT_CREDS);
	if (!e)
		return 0;
	e->euid = BPF_CORE_READ(newc, euid.val);
	e->arg0 = BPF_CORE_READ(newc, uid.val);
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/syscalls/sys_enter_connect")
int tp_connect(struct sys_enter_ctx *ctx)
{
	struct sockaddr_in_min sa = {};
	if (bpf_probe_read_user(&sa, sizeof(sa), (const void *)ctx->args[1]))
		return 0;
	if (sa.family != AF_INET && sa.family != AF_INET6)
		return 0;
	struct event *e = event_begin(EVT_CONNECT);
	if (!e)
		return 0;
	e->arg0 = ctx->args[0];
	e->arg1 = (__u16)((sa.port >> 8) | (sa.port << 8));
	if (sa.family == AF_INET6)
		e->flags |= FLAG_IPV6;
	else
		e->arg2 = sa.addr; /* raw network order; loader renders dotted quad into data */
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/syscalls/sys_enter_socket")
int tp_socket(struct sys_enter_ctx *ctx)
{
	struct event *e = event_begin(EVT_SOCKET);
	if (!e)
		return 0;
	e->arg0 = ctx->args[0];
	e->arg1 = ctx->args[1];
	bpf_ringbuf_submit(e, 0);
	return 0;
}

static __always_inline int emit_dup(struct sys_enter_ctx *ctx)
{
	struct event *e = event_begin(EVT_DUP);
	if (!e)
		return 0;
	e->arg0 = ctx->args[0];
	e->arg1 = ctx->args[1];
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/syscalls/sys_enter_dup2")
int tp_dup2(struct sys_enter_ctx *ctx) { return emit_dup(ctx); }

SEC("tp/syscalls/sys_enter_dup3")
int tp_dup3(struct sys_enter_ctx *ctx) { return emit_dup(ctx); }

/* Only emit opens of the dynamic-linker preload files; a full openat stream would be a flood. */
SEC("tp/syscalls/sys_enter_openat")
int tp_openat(struct sys_enter_ctx *ctx)
{
	char path[24] = {};
	bpf_probe_read_user_str(path, sizeof(path), (const void *)ctx->args[1]);
	if (!(path[0] == '/' && path[1] == 'e' && path[2] == 't' && path[3] == 'c' && path[4] == '/' &&
	      path[5] == 'l' && path[6] == 'd' && path[7] == '.' && path[8] == 's' && path[9] == 'o'))
		return 0;
	struct event *e = event_begin(EVT_OPEN);
	if (!e)
		return 0;
	bpf_probe_read_user_str(e->data, sizeof(e->data), (const void *)ctx->args[1]);
	e->arg0 = ctx->args[2]; /* open flags */
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/sched/sched_process_fork")
int tp_fork(struct sched_fork_ctx *ctx)
{
	struct event *e = event_begin(EVT_FORK);
	if (!e)
		return 0;
	e->arg0 = ctx->child_pid;
	bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tp/sched/sched_process_exit")
int tp_exit(struct sched_exit_ctx *ctx)
{
	struct event *e = event_begin(EVT_EXIT);
	if (!e)
		return 0;
	bpf_ringbuf_submit(e, 0);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
