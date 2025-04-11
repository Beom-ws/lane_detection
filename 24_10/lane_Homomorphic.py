    def homomorphic_filter(self, img): # 720 x 640
        img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y = img_YUV[:,:,0]
        rows, cols = y.shape
        
        y = cv2.resize(y, (512, 512))

        imgLog = np.log1p(np.array(y, dtype='float')/255)
        M, N = y.shape

        # Gaussian mask 생성
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        Xc, Yc = np.ceil(N/2), np.ceil(M/2)
        gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2
        
        # LPF, HPF 생성
        LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
        HPF = 1 - LPF
        
        LPF_shift = np.fft.ifftshift(LPF.copy())
        HPF_shift = np.fft.ifftshift(HPF.copy())
        
        img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
        img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
        img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분
        
        gamma1 = 0.3
        gamma2 = 1.2
        img_adjusting = gamma1 * img_LF + gamma2 * img_HF

        # 조정된 데이터를 exp 연산을 통해 이미지로 변환
        img_exp = np.expm1(img_adjusting) # exp(x) - 1
        img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1 사이로 정규화
        img_out = np.array(255 * img_exp, dtype='uint8') # 255를 곱해 intensity값 생성

        img_out = cv2.resize(img_out, (cols, rows))

        # YUV에서 Y 채널을 필터링된 이미지로 교체하고 RGB 공간으로 변환
        img_YUV[:,:,0] = img_out
        result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

        
        return result   
