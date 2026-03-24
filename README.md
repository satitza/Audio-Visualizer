# Audio-Visualizer
Audio Visualizer with python for terminal

Requirements: pip install PyAudioWPatch numpy psutil GPUtil

# Install
add this to your windows powershell profile

## 🎧 Soundbar Function (PowerShell)

```powershell
function soundbar {
    param(
        [switch]$stop,
        [switch]$full,
        [double]$size = 0.35,
        [int]$bars = 0,
        [int]$height = 0,
        [double]$gain = 0,
        [switch]$list
    )

    $script = "$env:USERPROFILE\scripts\audio_visualizer.py"

    if ($stop) {
        python $script --stop
    }
    elseif ($list) {
        python $script --list
    }
    elseif ($full) {
        # Full screen in current pane
        python $script --top @args
    }
    else {
        # Split bottom pane with visualizer (user keeps typing in top pane)
        $pyArgs = "--top"

        if ($bars -gt 0)   { $pyArgs += " --bars $bars" }
        if ($height -gt 0) { $pyArgs += " --height $height" }
        if ($gain -gt 0)   { $pyArgs += " --gain $gain" }

        wt split-pane -H -s $size -- python $script $pyArgs.Split(' ')
    }
}
```

---

### 🔥 Usage

```powershell
soundbar              # run in split pane
soundbar -full       # run fullscreen
soundbar -stop       # stop visualizer
soundbar -list       # list devices
soundbar -bars 50 -height 20 -gain 1.5
```



<img width="1893" height="1020" alt="image" src="https://github.com/user-attachments/assets/926bd50e-d3d8-4936-825f-131198ea2c30" />

<img width="1899" height="1026" alt="image" src="https://github.com/user-attachments/assets/7c4a2383-6e85-4b97-9970-d095e13d5922" />

