/*
 * Daemon.app's launcher.
 *
 * The .app exists only to be the grantable microphone (TCC) identity: a
 * launchd-spawned bare Python cannot obtain the grant, but a process started
 * from a signed bundle can, and macOS keys the grant to the bundle's code
 * identity — which survives `daemon update` because that only replaces the
 * daemon's Python env, never this binary.
 *
 * This MUST be a native universal2 Mach-O, never a shell script: a script main
 * executable makes LaunchServices launch the app as x86_64 (a Rosetta prompt)
 * and yields a weak TCC identity. A compiled launcher fixed both (spike fact 2).
 *
 * It is deliberately generic — identical for every install. The real daemon's
 * absolute path and subcommand arrive as argv from the LaunchAgent plist
 * (ProgramArguments = [this, <daemon-path>, run]) or from `open --args`
 * (<daemon-path> request-mic). So it just execs argv[1] with argv[1:]; the
 * exec'd process inherits this bundle's TCC identity (spike facts 4, 5).
 */
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "launcher: expected the daemon path as argv[1]\n");
        return 2;
    }
    execv(argv[1], &argv[1]);
    /* Only reached if execv failed. */
    perror("launcher: execv");
    return 1;
}
