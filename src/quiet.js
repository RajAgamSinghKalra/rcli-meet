#!/usr/bin/env node
// Launcher that strips the native addon's log spam.
//
// The RunAnywhere addon logs at INFO regardless of what the host asks for:
// rac_init() stores cfg.log_level into a variable used only by the core's
// internal logger, while the RAC_LOG_* macros check a separate min_level in
// rac_logger.cpp that defaults to INFO and never receives it. The bindings
// don't expose rac_logger_set_min_level either, so the only fix available
// without rebuilding the native addon is to filter its output here.
//
// stdin stays inherited so readline keeps the real TTY (typing still works);
// only stdout is piped through the filter.
const path = require('path');
const { spawn } = require('child_process');

const LOG_PREFIX = '[RAC]';

/** True if `s` is a (possibly partial) start of a log line. */
function couldBeLogLine(s) {
  return s.startsWith(LOG_PREFIX) || LOG_PREFIX.startsWith(s);
}

function createFilter(write) {
  let held = '';
  return {
    push(chunk) {
      const parts = (held + chunk).split('\n');
      // The last element has no newline yet -- it's either a log line still
      // arriving or live caption/answer text that must NOT be buffered
      // (captions repaint in place with no trailing newline, so holding them
      // would freeze the live display).
      held = parts.pop();
      for (const line of parts) {
        if (!line.startsWith(LOG_PREFIX)) write(line + '\n');
      }
      if (held && !couldBeLogLine(held)) {
        write(held);
        held = '';
      }
    },
    flush() {
      // Anything still held got held *because* it looked like a log line (push
      // releases everything else immediately), so at end-of-stream it's a
      // truncated log line -- dropping it beats emitting a stray "[RA".
      held = '';
    },
  };
}

function main() {
  const child = spawn(
    process.execPath,
    [path.join(__dirname, 'main.js'), ...process.argv.slice(2)],
    { stdio: ['inherit', 'pipe', 'ignore'] }
  );

  const filter = createFilter((text) => process.stdout.write(text));
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => filter.push(chunk));
  child.stdout.on('end', () => filter.flush());

  child.on('error', (err) => {
    console.log(`[rcli-meet] could not start: ${err.message}`);
    process.exit(1);
  });
  child.on('exit', (code, signal) => {
    filter.flush();
    process.exit(signal ? 1 : (code ?? 0));
  });

  // Let the child own Ctrl+C (it has the graceful-shutdown path); the console
  // delivers the signal to the whole process group anyway.
  process.on('SIGINT', () => {});
}

if (require.main === module) main();

module.exports = { createFilter };
