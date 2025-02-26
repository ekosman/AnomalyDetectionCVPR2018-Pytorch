#!/bin/bash

# Update package lists
apt-get update 

# Install necessary GUI libraries
apt-get install -y \
    libxkbcommon-x11-0 libxcb-xinerama0 libxcb-randr0 libxcb-shape0 \
    libxcb-glx0 libx11-xcb1 libqt5gui5 libqt5widgets5 libgtk-3-0 \
    qt5-gtk-platformtheme libpulse0

# Install PulseAudio for sound support
apt-get install -y libpulse0 pulseaudio libpulse-mainloop-glib0

# Install GStreamer plugins for video playback
apt-get install -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-alsa \
    gstreamer1.0-tools

# Update libstdc++ to ensure compatibility
conda install -c conda-forge libstdcxx-ng -y

echo "✅ Setup complete! Ready to run the demo."
