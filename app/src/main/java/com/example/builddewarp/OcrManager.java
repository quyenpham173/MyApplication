package com.example.builddewarp;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

class OcrManager {
    private static final String TESS_DATA = "/tessdata";
    private static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/Tess1";
    private String languge="vie";
    private static final String TAG = MainActivity.class.getSimpleName();

    String processOCR(Bitmap bitmap){
        TessBaseAPI tessBaseAPI = new TessBaseAPI();
        tessBaseAPI.init(DATA_PATH,languge);
        tessBaseAPI.setImage(bitmap);
        String s = tessBaseAPI.getUTF8Text().replace("_", "").replace("~", "");
        return s;
    }
    void prepareTessData(Context context){
        AssetManager assetManager= context.getAssets();
        InputStream in=null;
        OutputStream out=null;
        try{
            File dir = new File(DATA_PATH + TESS_DATA);
            if(!dir.exists()){
                dir.mkdirs();
            }
            String pathToDataFile = DATA_PATH+TESS_DATA+"/"+languge+".traineddata";
            if(!(new File(pathToDataFile)).exists()){
                in = assetManager.open("tessdata/"+languge+".traineddata");
                out = new FileOutputStream(pathToDataFile);
                byte [] buff = new byte[1024];
                int len ;
                while(( len = in.read(buff)) > 0){
                    out.write(buff,0,len);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
        }
        finally {
            if(in!=null) {
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(out!=null){
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
