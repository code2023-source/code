function out=GS(x,m,tau,theiler,nn)
%	Given a bivariate input data (x) calculates the cross-correlation and the 
%   non-linear interpendences S,H and N.
%	Results are in the vector out in the following order:
%	S(X|Y)， S(Y|X)， H(X|Y) ，H(Y|X)， N(X|Y) ，N(Y|X) ，cross-correlation.
%	Parameters to be set are: m (embedding dimension), tau (time delay), 
%	theiler (theiler correction in # data points) and NN (# of nearest neighbors).
%m=10;                   %embedding dimension
%tau=2;                  %time lag
%theiler=50;             %theiler correction
%nn = 10;                %number of nearest neighbors
ndatos = length(x);
xn=zscore(x);
cross = xcorr(xn(:,1),xn(:,2),0,'biased');%有偏的互相关函数
out=zeros(1,7);
sxy = 0; syx = 0; hxy = 0; hyx = 0; nxy = 0; nyx = 0;
for i = 1:ndatos-(m-1)*tau-1;
    for k=1:nn                                      %INICIALIZE AUX初始化AUX
       auxx(k) = 100000000;
       indexx(k) = 100000000;
       auxy(k) = 100000000;
       indexy(k) = 100000000;
    end 
    auxx(nn+1) = 0;
    auxy(nn+1) = 0;
    indexx(nn+1) = 100000000;
    indexy(nn+1) = 100000000;
    rrx = 0; rry = 0;
    for j = 1:ndatos-(m-1)*tau-1                    %*****************
          distx(j) = 0;
          disty(j) = 0;
          for k=0:m-1                                          %DISTANCES
            distx(j) = distx(j)+(x(i+k*tau,1)-x(j+k*tau,1)).^2;
            disty(j) = disty(j)+(x(i+k*tau,2)-x(j+k*tau,2)).^2;
          end 
          if ((abs(i-j)) > theiler)  
              if (distx(j) < auxx(1)) 
                 flagx=0;
                 for k=1:nn+1
                    if (distx(j) < auxx(k)) 
                       auxx(k) = auxx(k+1);
                       indexx(k) = indexx(k+1);
                      else
                       auxx(k-1) = distx(j);
                       indexx(k-1) = j;
                       flagx=1;
                    end 
                    if flagx==1;break;end
                 end
              end 
              if (disty(j) < auxy(1)) 
                 flagy=0;
                 for k=1:nn+1
                    if (disty(j) < auxy(k)) 
                       auxy(k) = auxy(k+1);
                       indexy(k) = indexy(k+1);
                      else
                       auxy(k-1) = disty(j);
                       indexy(k-1) = j;
                       flagy=1;
                    end 
                    if flagy==1,break,end
                 end 
              end           
          end     
          rrx = rrx + distx(j);        %SIZE OF THE ATTRACTORS
          rry = rry + disty(j);
    end                                             %***************** 
    rxx = 0; ryy = 0; rxy = 0; ryx = 0;
    for k=1:nn
         rxx = auxx(k) + rxx;
         ryy = auxy(k) + ryy;
         rxy = distx(indexy(k)) + rxy;
         ryx = disty(indexx(k)) + ryx;
    end 
    rxx = rxx/nn;
    ryy = ryy/nn;
    rxy = rxy/nn;
    ryx = ryx/nn;
    sxy = sxy + rxx/rxy;
    syx = syx + ryy/ryx;
    hxy =  hxy + log(rrx/((ndatos-(m-1)*tau-2)) / rxy);
    hyx =  hyx + log(rry/((ndatos-(m-1)*tau-2)) / ryx);
    nxy =  nxy + rxy / (rrx/((ndatos-(m-1)*tau-2))); 
    nyx =  nyx + ryx / (rry/((ndatos-(m-1)*tau-2)));  
end 
sxy = sxy/(ndatos-(m-1)*tau-1);
syx = syx/(ndatos-(m-1)*tau-1);
hxy = hxy/(ndatos-(m-1)*tau-1);
hyx = hyx/(ndatos-(m-1)*tau-1);
nxy = 1 - nxy/(ndatos-(m-1)*tau-1);
nyx = 1 - nyx/(ndatos-(m-1)*tau-1);
           
out(1)=sxy;
out(2)=syx;
out(3)=hxy;
out(4)=hyx;
out(5)=nxy;
out(6)=nyx;
out(7)=cross;

























