clc,clear
close all

% set DAVIS original video path
video_path='E:\DAVIS\';
% set your saving path
save_path='.\';
% load the mask
load('mask.mat')


resolution='480p/';
block_size=256;
compress_frame=8;

gt_save_path=[save_path,'gt/'];
meas_save_path=[save_path,'measurement/'];
if exist(gt_save_path,'dir')==0
   mkdir(gt_save_path);
end
if exist(meas_save_path,'dir')==0
   mkdir(meas_save_path);
end

num_yb=1;
name_obj=dir([video_path,'/JPEGImages/',resolution]);
for ii=3:length(name_obj)
   path=[video_path,'/JPEGImages/',resolution,name_obj(ii).name];
   name_frame=dir(path);
   pic1=imread([path,'/',name_frame(3).name]);
   w=size(pic1);
   x=1:block_size/2:w(1)-block_size;
   if x(end)<w(1)-block_size
        x=[x,w(1)-block_size];
   end
   
   y=1:block_size/2:w(2)-block_size;
   if y(end)<w(2)-block_size
        y=[y,w(2)-block_size];
   end

   
    if exist(gt_save_path,'dir')==0
       mkdir(gt_save_path);
    end
    if exist(meas_save_path,'dir')==0
       mkdir(meas_save_path);
    end
    load(['mask',num2str(compress_frame),'.mat'])
    save(strcat(save_path,'mask.mat'),'mask')
   
   for ll=3:compress_frame:length(name_frame)-compress_frame
       
       pic_block=zeros(size(pic1));
         for mm=1:compress_frame
             pic=imread([path,'/',name_frame(ll+mm-1).name]);
             pic=rgb2ycbcr(pic);
             pic_block(:,:,mm)=pic(:,:,1);
         end
         pic_block_mean=mean(pic_block,3);
         d_pic_block=pic_block-pic_block_mean;
         d_pic_block=d_pic_block.^2;
         pic_block_sigma=mean(d_pic_block,3);
         m=zeros(3,6);n=1;
         for i=1:3
            for j=1:6
                x1=pic_block_sigma(x(i):x(i)+block_size-1,y(j):y(j)+block_size-1,:);
                a1=max(x1(:))-min(x1(:));
                m(i,j)=a1;
                n=n+1;
            end
         end
         
         [a,index]=sort(m(:),'descend');
        
         for n=1:7
            x1=index(n); 
            i=mod((x1-1),3)+1;
            j=floor((x1-1)/3)+1;
            meas=zeros(block_size,block_size);
            patch_save=zeros(block_size,block_size,compress_frame);
            for mm=1:compress_frame
                pic=pic_block(x(i):x(i)+block_size-1,y(j):y(j)+block_size-1,mm);
                patch_save(:,:,mm)=pic;
                meas=meas+pic.*mask(:,:,mm);
            end
            save([gt_save_path,num2str(num_yb),'_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'],'patch_save');      
            save([meas_save_path,num2str(num_yb),'_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'],'meas');
            
            n1=[num2str(num_yb),'_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'];
            meas=zeros(block_size,block_size);
            p1=zeros(size(patch_save));
            for iii=1:256
                p1(:,iii,:)=patch_save(:,257-iii,:);
            end
            for mm=1:compress_frame
                p_1=p1(:,:,mm);
                meas=meas+p_1.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror','.mat'],'p1');      
            save([meas_save_path,n1(1:length(n1)-4),'_mirror','.mat'],'meas');

            meas=zeros(block_size,block_size);
            p2=permute(patch_save,[2,1,3]);
            for mm=1:compress_frame
                p_2=p2(:,:,mm);
                meas=meas+p_2.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_ori90','.mat'],'p2');      
            save([meas_save_path,n1(1:length(n1)-4),'_ori90','.mat'],'meas');

            meas=zeros(block_size,block_size);
            p3=permute(p1,[2,1,3]);
            for mm=1:compress_frame
                p_3=p3(:,:,mm);
                meas=meas+p_3.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror90','.mat'],'p3');      
            save([meas_save_path,n1(1:length(n1)-4),'_mirror90','.mat'],'meas');
            
            num_yb=num_yb+1;
         end

        

   end
  
end

x=1;
y=[1,128,250];
name_obj=dir([video_path,'/JPEGImages/',resolution]);
for ii=3:length(name_obj)
   path=[video_path,'/JPEGImages/',resolution,name_obj(ii).name];
   name_frame=dir(path);
   pic1=imread([path,'/',name_frame(3).name]);
   w=size(pic1);
   w=w(1:2);
   w=floor(0.6*w);

   
   for ll=3:compress_frame:length(name_frame)-compress_frame
       
       pic_block=zeros([w,3]);
         for mm=1:compress_frame
             pic=imread([path,'/',name_frame(ll+mm-1).name]);
             pic=rgb2ycbcr(pic);
             pic=imresize(pic,w,'bicubic');
            
             pic_block(:,:,mm)=pic(:,:,1);
         end
         pic_block_mean=mean(pic_block,3);
         d_pic_block=pic_block-pic_block_mean;
         d_pic_block=d_pic_block.^2;
         pic_block_sigma=mean(d_pic_block,3);
         m=zeros(1,3);n=1;
         for i=1:1
            for j=1:3
                x1=pic_block_sigma(x(i):x(i)+255,y(j):y(j)+255,:);
                a1=max(x1(:))-min(x1(:));
                m(i,j)=a1;
                n=n+1;
            end
         end
         
         [a,index]=sort(m(:),'descend');
         for n=1:2
            x1=index(n); 
            i=mod((x1-1),3)+1;
            j=floor((x1-1)/3)+1;
            meas=zeros(256,256);
            patch_save=zeros(256,256,compress_frame);
            for mm=1:compress_frame
                p1=pic_block(x(1):x(1)+255,y(j):y(j)+255,mm);
                patch_save(:,:,mm)=p1;
                meas=meas+p1.*mask(:,:,mm);
            end
            save([gt_save_path,num2str(num_yb),'_downsample_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'],'patch_save');      
            save([meas_save_path,num2str(num_yb),'_downsample_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'],'meas');
            
            n1=[num2str(num_yb),'_downsample_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'];
            meas=zeros(block_size,block_size);
            p1=zeros(size(patch_save));
            for iii=1:256
                p1(:,iii,:)=patch_save(:,257-iii,:);
            end
            for mm=1:compress_frame
                p_1=p1(:,:,mm);
                meas=meas+p_1.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror','.mat'],'p1');      
            save([meas_save_path,n1(1:length(n1)-4),'_mirror','.mat'],'meas');

            meas=zeros(block_size,block_size);
            p2=permute(patch_save,[2,1,3]);
            for mm=1:compress_frame
                p_2=p2(:,:,mm);
                meas=meas+p_2.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_ori90','.mat'],'p2');      
            save([meas_save_path,n1(1:length(n1)-4),'_ori90','.mat'],'meas');

            meas=zeros(block_size,block_size);
            p3=permute(p1,[2,1,3]);
            for mm=1:compress_frame
                p_3=p3(:,:,mm);
                meas=meas+p_3.*mask(:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror90','.mat'],'p3');      
            save([meas_save_path,n1(1:length(n1)-4),'_mirror90','.mat'],'meas');
            
            num_yb=num_yb+1;
         end
   end
  
end
