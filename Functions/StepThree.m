function Sout = StepThree(train_data,test_data,Sin,K,L,g)
%Step three of R2LMTL, fixed metric L and vector g, minimize over S
%Input:
%       train_data  -- Training Data
%       test_data   -- Testing Data
%       Sin         -- the input similarity matrix S
%       K           -- Number of metrics invovled
%       L           -- Learned metrics
%       g           -- vector g
%Output:
%       Sout        -- the output similarity matrix S

disp(['***Transductive Learning Phase!!!***'])

[M,Ntr] = size(train_data);
[M,Nte] = size(test_data);
NA = Ntr+Nte;
D = M-1;
train = train_data(1:D,:);
test = test_data(1:D,:);
X = [train,test];
C = 1;

Phi = zeros(NA,NA);
for k = 1:K
    eval(['L',num2str(k),' = L(:,(k-1)*D+1:D*k);']);
    eval(['Ltemp = L',num2str(k),';']);
    %To produce the Phi matrix
    Temp = zeros(NA,NA);
    for m = 1:NA
        for n = (m+1):NA
            Emn = (X(:,m)-X(:,n))'*Ltemp'*Ltemp*(X(:,m)-X(:,n));
            Temp(m,n) = Emn*g(k,m)*g(k,n)-C*max(0,1-Emn);
        end
    end
    Temp = Temp+Temp';
    %We add large number to the original place of Sin to avoid search this
    %area
    Temp(1:Ntr,1:Ntr) = 1e7*max(max(Temp));
    %Then we assign the temp to each Phi matrix of K
    Phi = Phi + Temp;
end

%Now we use greedy algorithm to produce the similarity matrix S
Sout = eye(NA,NA);
Sout(1:Ntr,1:Ntr) = Sin(1:Ntr,1:Ntr);
[Inx,Iny] = find(Phi < 0);
for i = 1:length(Inx)
    Sout(Inx(i),Iny(i)) = 1;
end
inm = 1:NA;
inm(Inx) = [];
for i = 1:length(inm)
    [tempx,tempy] = find(Phi(inm(i),:)==min(Phi(inm(i),:)));
    Sout(inm(i),tempy) = 1;
    Sout(tempy,inm(i)) = 1;
end