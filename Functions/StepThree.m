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
    eval(['Phi',num2str(k),' = Temp;']);
end

%Now we use greedy algorithm to produce the similarity matrix S
Sout = eye(NA,NA);
Sout(1:Ntr,1:Ntr) = Sin(1:Ntr,1:Ntr); 
for k = 1:K
    eval(['Phitemp = Phi',num2str(k),';']);
    [Inx,Iny] = find(Phitemp < 0);
    %We put these ares as 1
    for i = 1:length(Inx)
        Sout(Inx(i),Iny(i)) = 1;
    end
    %There are some rows that may not contain negative values for all the
    %metrics, we need to record these row indexes for each k
    inm = 1:NA;
    inm(Inx) = [];
    if k == 1
        inmf = inm;
    else
        inmf = intersect(inm,inmf);
    end
end

%let's see if we have crossover between these 
if isempty(inmf) ~= 1
    for m = 1:length(inmf)
        %inmf is the row index
        value = [];
        minindex = [];
        for k=1:K
            %for each metric, we will go over the column and find the
            %minium
            eval(['Phitemp = Phi',num2str(k),';']);
            %We need to find the minimum value for all the k
            indmin = find(Phitemp(inmf(m),:) == min(Phitemp(inmf(m),:)) & min(Phitemp(inmf(m),:)) >= 0);
            value = [value,Phitemp(inmf(m),indmin(1))];
            minindex = [minindex,indmin(1)];
        end
        [xx,yy] = min(value);
        indmin = minindex(yy(1));
        Sout(inmf(m),indmin) = 1;
        Sout(indmin,inmf(m)) = 1;
    end
end