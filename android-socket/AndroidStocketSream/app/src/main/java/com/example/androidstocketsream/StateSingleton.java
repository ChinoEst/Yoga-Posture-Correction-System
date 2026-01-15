package com.example.androidstocketsream;

import java.util.concurrent.atomic.AtomicInteger;

public class StateSingleton {

    // Singleton for the state management
    public static boolean waitInterval = false;  //If True, stop taking a photo for a while
    public static boolean runScanning = false;  //If True, respresent for be taking  pictures which are after first photo
    public static boolean first = true;  // If True,  taking a photo for bbox
    public static boolean started = false;
    public static boolean raw1_end = false;
    public static int difficult = 1; // 1:easy 2:medium 3:hard
    public static final String TAG = "AndroidSocketStream";


    // traicking numbers of thread
    private final AtomicInteger threadCount = new AtomicInteger(0);

    private SocketStream socketStream;

    private static StateSingleton INSTANCE = null;

    private StateSingleton() {};

    public static StateSingleton getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new StateSingleton();
        }
        return(INSTANCE);
    }

    public AtomicInteger getThreadCount() {
        return threadCount;
    }





    public void incrementThreadCount() {
        threadCount.incrementAndGet();
    }



    public int decrementThreadCount() {
        return threadCount.decrementAndGet();
    }






    public boolean areAllThreadsComplete() {
        return threadCount.get() == 0;
    }





    //set 0
    public void setThreadCount(int value) {
        threadCount.set(value);
    }



    public SocketStream getSocketStream() {
        return socketStream;
    }


    public void setSocketStream(SocketStream socketStream) {
        this.socketStream = socketStream;
    }


    public void reset() {
        first = true;
        runScanning = false;
        waitInterval = false;
        started = false;
        raw1_end = false;
        difficult = 1;


        threadCount.set(0);

        if (socketStream != null) {
            socketStream.clearImageList();
        }

        // clean callback or listener（avoid memory leak）
        if (socketStream != null) {
            socketStream.setOnImagesReadyCallback(null);
        }

        // reset Response
        socketStream.setSuccessResponse_bbox(false);
        socketStream.setSuccessResponse_connect(false);
        socketStream.setSuccessResponse3(false);
        socketStream.setResponse3_finish(false);
        //socketStream.setSuccessResponse2(false);

        first = true;
        runScanning = false;
    }

}
