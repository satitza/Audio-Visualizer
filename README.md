# Audio-Visualizer
Audio Visualizer with python for terminal

Requirements: pip install PyAudioWPatch numpy psutil GPUtil

# Install
add this to your windows powershell profile

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
    } elseif ($list) {
        python $script --list
    } elseif ($full) {
        # Full screen in current pane
        python $script --top @args
    } else {
        # Split bottom pane with visualizer (user keeps typing in top pane)
        $pyArgs = "--top"
        if ($bars -gt 0) { $pyArgs += " --bars $bars" }
        if ($height -gt 0) { $pyArgs += " --height $height" }
        if ($gain -gt 0) { $pyArgs += " --gain $gain" }
        wt split-pane -H -s $size -- python $script $pyArgs.Split(' ')
    }
}



<img width="1893" height="1020" alt="image" src="https://github.com/user-attachments/assets/926bd50e-d3d8-4936-825f-131198ea2c30" />
