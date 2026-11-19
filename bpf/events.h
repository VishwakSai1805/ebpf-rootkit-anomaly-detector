/* Shared kernel/userspace event ABI. Keep in sync with collector/events.py (SPEC.md §3). */
#ifndef EBPF_RK_EVENTS_H
#define EBPF_RK_EVENTS_H

#ifndef __VMLINUX_H__
#include <linux/types.h>
#endif

enum evt_type {
	EVT_EXECVE = 1,
	EVT_PTRACE = 2,
	EVT_COMMIT_CREDS = 3,
	EVT_CONNECT = 4,
	EVT_SOCKET = 5,
	EVT_DUP = 6,
	EVT_OPEN = 7,
	EVT_FORK = 8,
	EVT_EXIT = 9,
};

#define FLAG_LD_PRELOAD_ENV (1u << 0)
#define FLAG_IPV6           (1u << 1)

#define COMM_LEN 16
#define DATA_LEN 96

struct event {
	__u64 ts_ns;
	__u32 pid;
	__u32 tid;
	__u32 ppid;
	__u32 uid;
	__u32 euid;
	__u32 type;
	__s64 arg0;
	__s64 arg1;
	__s64 arg2;
	__u32 flags;
	__u32 pad;
	char comm[COMM_LEN];
	char data[DATA_LEN];
};

#endif
