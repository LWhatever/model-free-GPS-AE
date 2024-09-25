clear;close all;
RQ              = 6;                            %QAM order
order_types     = length(RQ);
rng(100);
Normalize_flag  = 0;                                %   1 power normalization，0 amplitude normalization

%% QAM映射
for n1 = 1:order_types
    BitsPerSymbol       = RQ(n1);
    QAMorder            = 2^BitsPerSymbol; 
    
    if QAMorder == 1
        dec_data  = [0,1].';
    else
        dec_data = (0:QAMorder-1)';
        BitData = de2bi(dec_data,BitsPerSymbol,'left-msb');
        
        save(['./Init/BitData_QAM',num2str(QAMorder),'.txt'],'BitData','-ascii');
    end
    % QAM modulation
    if QAMorder == 8
        qamdata(n1,:) = qammod_diamond8(dec_data);% 8阶diamond星座点
    elseif  QAMorder == 1
        qamdata(n1,:) = dec_data(n1,:);
    else
        qamdata(n1,:) = qammod(dec_data,QAMorder);% regular QAM
    end
    
    
    % draw constellation
    for i = 1:QAMorder
        [row_tmp,col_tmp] = find(dec_data == i-1);
        sym_index(i,:) = [row_tmp(1), col_tmp(1)];
        qamdata_single(i) = qamdata(n1,sym_index(i));
        figure(n1);
%         text(real(qamdata_single(i)),imag(qamdata_single(i)),['   ',num2str(BitData(row_tmp(1),(col_tmp(1)-1)*BitsPerSymbol+1:col_tmp(1)*BitsPerSymbol))],'FontSize',9,'FontWeight','bold');
        hold on;
        plot(real(qamdata_single(i)),imag(qamdata_single(i)),'o','LineWidth',5,'MarkerSize',14,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,1]);
    end
    
    % calculate the average power of the QAM signal
    if Normalize_flag == 1
        if QAMorder == 8
            AVT(n1,:) = sqrt(mean(abs(qammod_diamond8(0:QAMorder-1)).^2)); 
        elseif QAMorder == 1
            AVT(n1,:) = ones(1,datano);
        else
            AVT(n1,:) = sqrt(mean(abs(qammod(0:QAMorder-1,QAMorder)).^2));    
        end
    else
        if QAMorder == 1
            AVT(n1,:) = ones(1,datano);
        else
            AVT(n1,:) = max(abs(qamdata(n1,:)));
        end
    end
    qamdata(n1,:) = qamdata(n1,:)./AVT(n1,:);
    qamdata_n1    = qamdata(n1,:).';
    
    qamdata_I_forNN = real(qamdata_n1);
    qamdata_Q_forNN = imag(qamdata_n1);
    qamdata_I_forNN = qamdata_I_forNN./max(abs(qamdata_I_forNN),[],'all');
    qamdata_Q_forNN = qamdata_Q_forNN./max(abs(qamdata_Q_forNN),[],'all');
    save(['./Init/QAM',num2str(QAMorder),'_I.txt'],'qamdata_I_forNN', '-ASCII');
    save(['./Init/QAM',num2str(QAMorder),'_Q.txt'],'qamdata_Q_forNN', '-ASCII');
    figure;plot(qamdata_I_forNN,qamdata_Q_forNN,'.');
end
figure(n1);
set(gcf,'unit','normalized','position',[0.0,0.03,0.5,0.75]);
box on;
%% generate normalized QAM data
save('./NN/demodulationfile.mat','qamdata','AVT');

%% NN out
python_flag = 0;
if python_flag == 1
    system('conda activate tensorflow && python A4_0_TransInit.py');
end
Eecoder_sym = load(['./NN/Encoder_Init_dataout_mmWave_',num2str(RQ),'bits.txt']);
[C,L] = size(Eecoder_sym);
Eecoder_sym_I   = Eecoder_sym(:,1:L/2);
Eecoder_sym_Q   = Eecoder_sym(:,L/2+1:end);
Eecoder_sym_IQ  = complex(Eecoder_sym_I,Eecoder_sym_Q);

noise       = wgn(C,L,0.00,'linear');
Eecoder_sym_noise = Eecoder_sym + noise;
Eecoder_sym_I_noise   = Eecoder_sym_noise(:,1:L/2);
Eecoder_sym_Q_noise   = Eecoder_sym_noise(:,L/2+1:end);
Eecoder_sym_IQ_noise  = complex(Eecoder_sym_I_noise,Eecoder_sym_Q_noise);

% figure;plot(Eecoder_sym_IQ_noise,'.');hold on;
figure;hold on;
sym_index = zeros(QAMorder,2);
Eecoder_sym_IQ_single = zeros(QAMorder,1);
for i = 1:QAMorder
    [row_tmp,col_tmp] = find(dec_data == i-1);
    sym_index(i,:) = [row_tmp(1), col_tmp(1)];
    Eecoder_sym_IQ_single(i) = Eecoder_sym_IQ(sym_index(i));
%     text(real(Eecoder_sym_IQ_single(i)),imag(Eecoder_sym_IQ_single(i)),['   ',num2str(BitData_rs(row_tmp(1),(col_tmp(1)-1)*BitsPerSymbol+1:col_tmp(1)*BitsPerSymbol))],'FontSize',9,'FontWeight','bold');
    plot(Eecoder_sym_IQ_single(i),'o','LineWidth',5,'MarkerSize',14,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,1]);axis([-1.2 1.2 -1.2 1.2]);
end
set(gcf,'unit','normalized','position',[0.0,0.03,0.5,0.75]);
box on;

% Save figures
f = gcf;
figure_num = f.Number;
for i = 1:figure_num
    fig = figure(i);
    exportgraphics(fig, sprintf('./Pics/A4_0_Figure%d.png', i), 'Resolution', 300);
end