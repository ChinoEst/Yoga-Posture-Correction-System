package com.example.androidstocketsream;

import android.content.Context;
import android.media.MediaPlayer;

public class VoiceHandler {
    private MediaPlayer mediaPlayer;
    private Context context;

    public VoiceHandler(Context context) {
        this.context = context.getApplicationContext();
    }


    public void playAudioByNumber(int number) {
        if (mediaPlayer != null) {
            mediaPlayer.release(); // 


        String resourceName = "raw" + number; 
        int resId = context.getResources().getIdentifier(resourceName, "raw", context.getPackageName());

        if (resId == 0) {
            
            throw new IllegalArgumentException("useless raw：" + resourceName);
        }

        mediaPlayer = MediaPlayer.create(context, resId);

        if (mediaPlayer != null) {
            mediaPlayer.setOnCompletionListener(mp -> {
                mediaPlayer.release(); 
                mediaPlayer = null;
            });
            mediaPlayer.start();
        }
    }

    //stop
    public void stopAudio() {
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            mediaPlayer.stop();
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }

    //release
    public void release() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }
}
