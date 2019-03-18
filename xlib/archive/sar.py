
def sar_proc(idx):
    global sc_pos
    global data
    global aux
    global corr_window 
    global track

    p=prog.Prog(len(idx))
    out=[]
    for rec in idx:
        p.print_Prog(int(rec)) 
        r_i=sc_pos[rec][0:3]
        pri_code=data[rec]['PULSE_REPETITION_INTERVAL']               
                
        if pri_code == 1: pri=1428E-6
        elif pri_code == 2: pri=1429E-6
        elif pri_code == 3: pri=1290E-6
        elif pri_code == 4: pri=2856E-6
        elif pri_code == 5: pri=2984E-6
        elif pri_code == 6: pri=2580E-6 
        else: pri=0
        tx0=data[rec]['RECEIVE_WINDOW_OPENING_TIME']*0.0375E-6+pri-11.98E-6#-2E-6      
        
        # Process pules   
        tof=tx0/0.0375E-6#+delta
        
        et=np.zeros(corr_window+1)
        phase=np.zeros(corr_window+1)
        signal=np.zeros(corr_window+1,dtype=np.complex)
        alongtrack=np.zeros(corr_window+1)
        record=np.zeros((corr_window+1,2048))
        
        j=0
        for n in range(int(rec-corr_window/2),int(rec+corr_window/2+1)):  
            # Get auxillary data
            alongtrack[j]=data[n]['TLP']  
            et[j]=aux[n]['EPHEMERIS_TIME']
            #pri_code=bs.BitArray('0b'+
            #                    bs.BitArray(
            #                    uint=data[n]['PULSE_REPETITION_INTERVAL'],
            #                    length=8).bin[0:4]).uint
            pri_code=data[rec]['PULSE_REPETITION_INTERVAL']                   
            if pri_code is 1: pri=1428E-6
            elif pri_code is 2: pri=1429E-6
            elif pri_code is 3: pri=1290E-6
            elif pri_code is 4: pri=2856E-6
            elif pri_code is 5: pri=2984E-6
            elif pri_code is 6: pri=2580E-6 
            else: pri=0
            tx=data[n]['RECEIVE_WINDOW_OPENING_TIME']*0.0375E-6+pri-11.98E-6#-2E-6 
            #Compute phase shift
            d=spice.vnorm(sc_pos[n][0:3]-r_i)
            phase[j]=d*2000/c/0.0375E-6+tof-tx/0.0375E-6
            for delta in range(0,2048):
                sample=int(tof+delta-(tx/0.0375E-6-phase[j]))
                if sample>=0 and sample<2048:
                    signal[j]=track[sample] 
                    zero_doppler=np.fft.fft(signal)
                    record[j,delta]=np.sqrt(zero_doppler[0].real**2
                                           +zero_doppler[0].imag**2)
            j+=1
        dist=(np.argmax(np.mean(record,axis=0))+tof)*0.0375E-6*c/2000
        h=data[rec]['RADIUS_N']-3389-dist
        out.append(np.array([rec,h]))

    #q.put(out)
    return out
    plt.show()
