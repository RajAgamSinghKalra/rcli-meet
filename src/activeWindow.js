// Best-effort "what app/window was being recorded" for save-folder naming.
// Windows-only; falls back quietly everywhere else and on any failure --
// this is a label, not something worth crashing a session over.
const { execFileSync } = require('child_process');

const PS_SCRIPT = `
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class RcliMeetWin32 {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
}
"@
$h = [RcliMeetWin32]::GetForegroundWindow()
$sb = New-Object System.Text.StringBuilder 256
[RcliMeetWin32]::GetWindowText($h, $sb, 256) | Out-Null
$sb.ToString()
`;

function getActiveWindowTitle() {
  try {
    const out = execFileSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', PS_SCRIPT], {
      timeout: 3000,
      windowsHide: true,
    });
    const title = out.toString().trim();
    return title || 'unknown-app';
  } catch {
    return 'unknown-app';
  }
}

module.exports = { getActiveWindowTitle };
