// SPDX-License-Identifier: GPL-2.0
/* Minimal libbpf loader: attaches all programs in probes.bpf.o and prints one JSON event per line. */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <arpa/inet.h>
#include <bpf/libbpf.h>
#include "events.h"

static volatile sig_atomic_t stop;
static void on_sig(int s) { (void)s; stop = 1; }

static const char *tname(__u32 t)
{
	switch (t) {
	case EVT_EXECVE: return "execve";
	case EVT_PTRACE: return "ptrace";
	case EVT_COMMIT_CREDS: return "commit_creds";
	case EVT_CONNECT: return "connect";
	case EVT_SOCKET: return "socket";
	case EVT_DUP: return "dup";
	case EVT_OPEN: return "open";
	case EVT_FORK: return "fork";
	case EVT_EXIT: return "exit";
	}
	return "unknown";
}

static void json_str(const char *s, size_t max)
{
	putchar('"');
	for (size_t i = 0; i < max && s[i]; i++) {
		unsigned char c = (unsigned char)s[i];
		if (c == '"' || c == '\\') printf("\\%c", c);
		else if (c < 0x20 || c > 0x7e) printf("\\u%04x", c);
		else putchar(c);
	}
	putchar('"');
}

static int handle(void *ctx, void *data, size_t len)
{
	const struct event *e = data;
	char ip[INET_ADDRSTRLEN] = "";
	(void)ctx;
	if (len < sizeof(*e)) return 0;
	if (e->type == EVT_CONNECT && !(e->flags & FLAG_IPV6)) {
		struct in_addr a = { .s_addr = (in_addr_t)e->arg2 };
		inet_ntop(AF_INET, &a, ip, sizeof(ip));
	}
	printf("{\"ts_ns\":%llu,\"pid\":%u,\"tid\":%u,\"ppid\":%u,\"uid\":%u,\"euid\":%u,\"type\":\"%s\","
	       "\"arg0\":%lld,\"arg1\":%lld,\"arg2\":%lld,\"flags\":%u,\"comm\":",
	       (unsigned long long)e->ts_ns, e->pid, e->tid, e->ppid, e->uid, e->euid, tname(e->type),
	       (long long)e->arg0, (long long)e->arg1, (long long)(*ip ? 0 : e->arg2), e->flags);
	json_str(e->comm, COMM_LEN);
	printf(",\"data\":");
	json_str(*ip ? ip : e->data, DATA_LEN);
	printf("}\n");
	fflush(stdout);
	return 0;
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "bpf/build/probes.bpf.o";
	struct bpf_object *obj = bpf_object__open_file(path, NULL);
	if (!obj) { fprintf(stderr, "open %s failed\n", path); return 2; }
	if (bpf_object__load(obj)) { fprintf(stderr, "load failed (need CAP_BPF/CAP_SYS_ADMIN + BTF)\n"); return 3; }

	struct bpf_program *prog;
	int attached = 0;
	bpf_object__for_each_program(prog, obj) {
		if (bpf_program__attach(prog))
			attached++;
		else
			fprintf(stderr, "warn: attach %s failed\n", bpf_program__name(prog));
	}
	if (!attached) { fprintf(stderr, "no program could be attached (tracefs/perf unavailable?)\n"); return 5; }
	struct bpf_map *m = bpf_object__find_map_by_name(obj, "events");
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(m), handle, NULL, NULL);
	if (!rb) { fprintf(stderr, "ringbuf failed\n"); return 4; }
	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);
	while (!stop) {
		int n = ring_buffer__poll(rb, 100);
		if (n < 0 && n != -4 /* EINTR */) break;
	}
	ring_buffer__free(rb);
	bpf_object__close(obj);
	return 0;
}
