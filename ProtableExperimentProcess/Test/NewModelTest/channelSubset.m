function subslayers=channelSubset(subsets,index)
    N=numel(subsets);
     
    for i=1:N
     subslayers(i)=functionLayer(@(z) z(:,subsets(i),:,:),'Name',"Group"+index+"_"+i); %#ok<AGROW> 
    end
end
